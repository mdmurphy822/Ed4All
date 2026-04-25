"""
HTML Content Parser

Extracts structured content from Courseforge-generated HTML modules.
Supports two metadata tiers from Courseforge output:
  1. JSON-LD blocks (<script type="application/ld+json">) — structured page metadata
  2. data-cf-* attributes — inline per-element metadata
Falls back to regex heuristics for non-Courseforge IMSCC packages.
"""

import json as json_mod
import re
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable so lib.ontology.bloom resolves when
# this module is executed from inside Trainforge/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.ontology.bloom import detect_bloom_level as _canonical_detect_bloom_level  # noqa: E402
from lib.ontology.bloom import get_verbs_list as _get_canonical_verbs_list  # noqa: E402


@dataclass
class ContentSection:
    """A section of content from an HTML module."""
    heading: str
    level: int  # h1=1, h2=2, etc.
    content: str
    word_count: int
    components: List[str] = field(default_factory=list)  # flip-card, accordion, etc.
    content_type: Optional[str] = None  # from data-cf-content-type
    key_terms: List[str] = field(default_factory=list)  # from data-cf-key-terms
    # REC-VOC-02 (Wave 2, Worker K): deterministic teaching_role emitted by
    # Courseforge on flip-card/self-check/activity elements. When a section
    # contains exactly one distinct data-cf-teaching-role value among its
    # tagged children, ``teaching_role`` surfaces it; if multiple distinct
    # values appear the field stays None and the consumer should fall back
    # to the JSON-LD ``teachingRole`` array or the LLM classifier.
    # ``teaching_roles`` always lists every distinct value seen for audit.
    teaching_role: Optional[str] = None
    teaching_roles: List[str] = field(default_factory=list)
    # REC-JSL-03 (Wave 3, Worker M): learning-objective references harvested
    # from ``data-cf-objective-ref`` attributes on ``.activity-card`` and
    # ``.self-check`` elements within the section body. Courseforge emits
    # these at generate_course.py:378,491. Multiple activities per section
    # may cite different LOs; the list holds distinct values sorted
    # deterministically. Downstream consumers (process_course._create_chunk)
    # merge these into a chunk's ``learning_outcome_refs`` so the
    # Activity→LO KG edge materializes.
    objective_refs: List[str] = field(default_factory=list)
    # Wave 10: ``data-cf-source-ids`` values harvested from the section body.
    # Courseforge emits these on ``<section>`` / heading / component wrapper
    # elements per Wave 9 (P2 decision: never on ``<p>``/``<li>``/``<tr>``).
    # Stored as the raw ``sourceId`` strings (``dart:{slug}#{block_id}``);
    # process_course.py converts them to full SourceReference dicts with an
    # auto-role of ``contributing`` when JSON-LD doesn't supply the full
    # shape. Sorted + deduplicated for deterministic downstream diffs.
    source_references: List[str] = field(default_factory=list)


@dataclass
class LearningObjective:
    """A learning objective extracted from HTML content."""
    id: Optional[str]
    text: str
    bloom_level: Optional[str] = None
    bloom_verb: Optional[str] = None
    cognitive_domain: Optional[str] = None  # factual/conceptual/procedural/metacognitive
    key_concepts: List[str] = field(default_factory=list)
    assessment_suggestions: List[str] = field(default_factory=list)
    # Wave 59 (Courseforge emit) / Wave 69 (Trainforge consume): LO hierarchy
    # tier derived from canonical ID prefix. ``terminal`` = course-wide
    # rollup (TO-NN); ``chapter`` = chapter-level LO (CO-NN) rolling up to
    # a terminal. Elided when the JSON-LD doesn't declare it (legacy pre-
    # Wave 59 corpus).
    hierarchy_level: Optional[str] = None
    # Wave 59 (Courseforge emit) / Wave 69 (Trainforge consume): parent LO
    # ID — the terminal objective a chapter LO rolls up to. Absent on
    # terminals (they are KG roots). Optional on chapter LOs — carried when
    # Courseforge's synthesized_objectives.json supplied the mapping.
    parent_objective_id: Optional[str] = None
    # Wave 57 (Courseforge emit) / Wave 69 (Trainforge consume): Bloom-
    # qualified LO→concept edges. Each entry is {"concept": <slug>,
    # "bloom_level": <canonical level>} — note the snake_case keys (our
    # internal convention) vs. Courseforge's camelCase
    # targetedConcepts/bloomLevel on the wire. Bloom levels are lowercased
    # at parse time to match Trainforge's case-insensitive reference
    # resolution. Fed into build_semantic_graph to materialize the Wave 66
    # ``targets-concept`` edge type.
    targeted_concepts: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ParsedHTMLModule:
    """Parsed HTML module structure."""
    title: str
    word_count: int
    sections: List[ContentSection] = field(default_factory=list)
    learning_objectives: List[LearningObjective] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    interactive_components: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # New fields populated from JSON-LD / data-cf-* attributes
    page_id: Optional[str] = None
    misconceptions: List[Dict[str, str]] = field(default_factory=list)
    prerequisite_pages: List[str] = field(default_factory=list)
    suggested_assessment_types: List[str] = field(default_factory=list)
    # REC-JSL-03 (Wave 3, Worker M): page-level union of every distinct
    # ``data-cf-objective-ref`` value found anywhere in the HTML. Used as
    # the fallback attachment set in process_course when a chunk cannot be
    # mapped back to a specific section (the no-sections code path in
    # _chunk_content). Populated even when ``sections`` is empty.
    objective_refs: List[str] = field(default_factory=list)
    # Wave 10: page-level aggregated source references. Each entry is a
    # full ``SourceReference`` dict (per schemas/knowledge/source_reference
    # .schema.json) — ``{sourceId, role, ...}``. Precedence:
    #   1. JSON-LD ``sourceReferences`` (page-level + section-level) copied
    #      verbatim (full shape when Courseforge is Wave 9+).
    #   2. ``data-cf-source-ids`` HTML attributes (stringified sourceId
    #      only) synthesised as ``{sourceId, role: 'contributing'}`` when
    #      the sourceId isn't already represented in the JSON-LD set.
    # Deduped by sourceId; first-seen wins on role collision so JSON-LD's
    # authoritative role (primary / contributing / corroborating) is
    # preserved over the HTML-attr fallback's default 'contributing'.
    source_references: List[Dict[str, Any]] = field(default_factory=list)


class HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML.

    Skips:
      - ``<script>`` and ``<style>`` subtrees (always).
      - Any subtree rooted at an element carrying ``data-cf-role="template-chrome"``
        (Worker Q). Courseforge marks repeated page chrome — header, footer,
        skip link — with that attribute so the chunk text field doesn't
        contain boilerplate that every page duplicates. The n-gram boilerplate
        detector in ``Trainforge/rag/boilerplate_detector.py`` stays as
        belt-and-suspenders for non-Courseforge IMSCC.
    """

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.current_tag = None
        self.in_script = False
        self.in_style = False
        # Worker Q: count of currently-open template-chrome ancestors. When
        # nonzero, text data is discarded.
        self._template_chrome_depth = 0

    def _is_template_chrome(self, attrs) -> bool:
        for name, value in attrs:
            if name == "data-cf-role" and value == "template-chrome":
                return True
        return False

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag == 'script':
            self.in_script = True
        elif tag == 'style':
            self.in_style = True
        if self._is_template_chrome(attrs):
            self._template_chrome_depth += 1

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_script = False
        elif tag == 'style':
            self.in_style = False
        # Close template-chrome scope when we see the matching end tag for
        # a chrome-flagged element. html.parser doesn't give us the attrs on
        # endtag, so we use a heuristic: template chrome is only emitted on
        # a known small set of tags (`header`, `footer`, `a.skip-link`).
        # The counter decrements on those tag names when we're inside a
        # chrome region. For robustness this matches any end tag that
        # corresponds to a currently-open chrome region.
        if self._template_chrome_depth > 0 and tag in _CHROME_TAGS:
            self._template_chrome_depth -= 1
        self.current_tag = None

    def handle_startendtag(self, tag, attrs):
        # Self-closing chrome elements (rare but possible, e.g., <br data-cf-role="template-chrome"/>)
        # shouldn't leave the counter incremented.
        if tag == 'script':
            self.in_script = True
            self.in_script = False
        elif tag == 'style':
            self.in_style = True
            self.in_style = False
        # Chrome self-closers are transient — no effect on depth.

    def handle_data(self, data):
        if self.in_script or self.in_style:
            return
        if self._template_chrome_depth > 0:
            return
        text = data.strip()
        if text:
            self.text_parts.append(text)

    def get_text(self) -> str:
        return ' '.join(self.text_parts)


# Tags that Courseforge's generate_course.py emits with
# ``data-cf-role="template-chrome"``. Keeping this narrow avoids
# under-counting end tags in complex nested chrome.
_CHROME_TAGS = {"header", "footer", "a", "div", "nav", "aside"}


class HTMLContentParser:
    """
    Parser for Courseforge-generated HTML content.

    Usage:
        parser = HTMLContentParser()
        module = parser.parse(html_content)
        print(f"Word count: {module.word_count}")
        for obj in module.learning_objectives:
            print(f"LO: {obj.text}")
    """

    # Bloom's taxonomy verbs by level.
    # Source of truth: schemas/taxonomies/bloom_verbs.json (loaded via
    # lib.ontology.bloom). Migrated in Wave 1.2 / Worker H (REC-BL-01).
    BLOOM_VERBS = _get_canonical_verbs_list()

    # Interactive component patterns
    COMPONENT_PATTERNS = {
        "flip-card": r'class="[^"]*flip-card[^"]*"',
        "accordion": r'class="[^"]*accordion[^"]*"',
        "tabs": r'class="[^"]*nav-tabs[^"]*"',
        "callout": r'class="[^"]*(?:callout|alert)[^"]*"',
        "knowledge-check": r'class="[^"]*knowledge-check[^"]*"',
        "activity-card": r'class="[^"]*activity-card[^"]*"'
    }

    def parse(self, html_content: str) -> ParsedHTMLModule:
        """
        Parse HTML content into structured format.

        Extraction priority: JSON-LD > data-cf-* attributes > regex heuristics.

        Args:
            html_content: HTML string to parse

        Returns:
            ParsedHTMLModule with extracted structure
        """
        # Extract text
        extractor = HTMLTextExtractor()
        extractor.feed(html_content)
        text = extractor.get_text()
        word_count = len(text.split())

        # Extract JSON-LD metadata (highest fidelity, from Courseforge output)
        json_ld = self._extract_json_ld(html_content)

        # Extract title
        title = self._extract_title(html_content)

        # Extract sections (with data-cf-* attribute support)
        sections = self._extract_sections(html_content)

        # Extract learning objectives (JSON-LD > data-attr > regex)
        objectives = self._extract_objectives(html_content, json_ld)

        # Extract key concepts
        concepts = self._extract_concepts(html_content)

        # Detect interactive components
        components = self._detect_components(html_content)

        # Build metadata dict
        metadata: Dict[str, Any] = {}
        if json_ld:
            metadata["courseforge"] = json_ld

        # Extract page-level fields from JSON-LD
        page_id = json_ld.get("pageId") if json_ld else None
        raw_misconceptions = json_ld.get("misconceptions", []) if json_ld else []
        # Wave 60 (Courseforge emit) / Wave 69 (Trainforge consume): normalize
        # Misconception dicts from JSON-LD camelCase (bloomLevel /
        # cognitiveDomain) to Trainforge snake_case (bloom_level /
        # cognitive_domain) and lowercase the bloom level. Only the canonical
        # required pair (misconception + correction) is mandatory; bloom /
        # domain are optional and silently absent on pre-Wave-60 corpora.
        misconceptions: List[Dict[str, Any]] = []
        for mc in raw_misconceptions:
            if not isinstance(mc, dict):
                # Pass non-dict entries through unchanged (strings etc.)
                misconceptions.append(mc)
                continue
            entry: Dict[str, Any] = {}
            statement = mc.get("misconception")
            if isinstance(statement, str):
                entry["misconception"] = statement
            correction = mc.get("correction")
            if isinstance(correction, str):
                entry["correction"] = correction
            # Preserve legacy fields if present (concept_id, lo_id etc.)
            for k, v in mc.items():
                if k in ("misconception", "correction", "bloomLevel",
                         "cognitiveDomain", "bloom_level", "cognitive_domain"):
                    continue
                entry[k] = v
            bloom = mc.get("bloomLevel") or mc.get("bloom_level")
            if isinstance(bloom, str) and bloom:
                entry["bloom_level"] = bloom.lower()
            domain = mc.get("cognitiveDomain") or mc.get("cognitive_domain")
            if isinstance(domain, str) and domain:
                entry["cognitive_domain"] = domain
            misconceptions.append(entry)

        # Wave 81 (Worker C): bridging fallback for HTML-attr-only emit.
        # Wave 79 content-generator subagents tag the misconception
        # paragraph with ``data-cf-misconception="true"`` but don't always
        # populate JSON-LD ``misconceptions[]``. Forward fix: the
        # ``Courseforge/templates/chunk_templates.md`` Template 3 spec now
        # mandates dual-emit. Backward bridge: scan the HTML for
        # ``data-cf-misconception="true"`` paragraphs whose text isn't
        # already covered by a JSON-LD entry, extract a misconception (with
        # the sibling "right approach" / "correct approach" paragraph as
        # the correction), and append. JSON-LD wins on text equality so
        # the bridge never produces duplicates.
        misconceptions.extend(
            self._extract_misconceptions_from_attrs(
                html_content, existing=misconceptions
            )
        )
        prerequisite_pages = json_ld.get("prerequisitePages", []) if json_ld else []
        suggested_assessments = json_ld.get("suggestedAssessmentTypes", []) if json_ld else []

        # REC-JSL-03 (Wave 3, Worker M): page-level union of every distinct
        # data-cf-objective-ref in the raw HTML. Covers activities/self-checks
        # that live outside any section (e.g., pages without headings) so the
        # no-sections chunk code path in process_course still materializes
        # the Activity→LO KG edge.
        page_obj_ref_matches = re.findall(
            r'data-cf-objective-ref="([^"]*)"', html_content
        )
        page_obj_refs = sorted({r for r in page_obj_ref_matches if r})

        # Wave 10: page-level source_references aggregated with precedence
        # JSON-LD (full shape) > data-cf-source-ids (sourceId strings
        # auto-roled as 'contributing'). First-seen wins on sourceId
        # collision so JSON-LD's authoritative role is preserved.
        page_source_refs = self._build_page_source_refs(
            json_ld, sections, html_content
        )

        return ParsedHTMLModule(
            title=title,
            word_count=word_count,
            sections=sections,
            learning_objectives=objectives,
            key_concepts=concepts,
            interactive_components=components,
            metadata=metadata,
            page_id=page_id,
            misconceptions=misconceptions,
            prerequisite_pages=prerequisite_pages,
            suggested_assessment_types=suggested_assessments,
            objective_refs=page_obj_refs,
            source_references=page_source_refs,
        )

    def _build_page_source_refs(
        self,
        json_ld: Optional[Dict[str, Any]],
        sections: List[ContentSection],
        html_content: str,
    ) -> List[Dict[str, Any]]:
        """Wave 10: Aggregate page-level source_references with precedence.

        Precedence:
          1. JSON-LD page-level ``sourceReferences`` (full SourceReference
             shape — sourceId, role, optional weight/confidence/pages/
             extractor) copied verbatim.
          2. JSON-LD section-level ``sourceReferences`` (same shape) —
             appended after page-level.
          3. ``data-cf-source-ids`` values from HTML attributes (strings
             only) synthesised as ``{sourceId, role: 'contributing'}`` and
             appended last.

        First-seen wins on sourceId collision so JSON-LD's authoritative
        role survives over the HTML-attr fallback. Returns an empty list
        when no refs are found (pre-Wave-9 corpus) — consumers treat
        absence as "unknown", never an error.
        """
        refs: List[Dict[str, Any]] = []
        seen: set = set()

        def _add(entry: Dict[str, Any]) -> None:
            sid = entry.get("sourceId")
            if not isinstance(sid, str) or not sid:
                return
            if sid in seen:
                return
            seen.add(sid)
            refs.append(dict(entry))

        # 1. Page-level JSON-LD sourceReferences
        if isinstance(json_ld, dict):
            for entry in json_ld.get("sourceReferences", []) or []:
                if isinstance(entry, dict):
                    _add(entry)
            # 2. Section-level JSON-LD sourceReferences
            for sec in json_ld.get("sections", []) or []:
                if not isinstance(sec, dict):
                    continue
                for entry in sec.get("sourceReferences", []) or []:
                    if isinstance(entry, dict):
                        _add(entry)

        # 3. HTML data-cf-source-ids fallback — synthesised 'contributing'
        all_html_ids: List[str] = []
        for raw in re.findall(r'data-cf-source-ids="([^"]*)"', html_content):
            for piece in raw.split(","):
                piece = piece.strip()
                if piece:
                    all_html_ids.append(piece)
        for sid in all_html_ids:
            _add({"sourceId": sid, "role": "contributing"})

        return refs

    def _extract_json_ld(self, html: str) -> Optional[Dict[str, Any]]:
        """Extract the first JSON-LD block with Courseforge context from HTML."""
        pattern = r'<script\s+type="application/ld\+json"[^>]*>(.*?)</script>'
        for match in re.finditer(pattern, html, re.DOTALL | re.IGNORECASE):
            try:
                data = json_mod.loads(match.group(1))
                # Accept any JSON-LD block, prefer Courseforge-specific ones
                if isinstance(data, dict):
                    return data
            except (json_mod.JSONDecodeError, ValueError):
                continue
        return None

    def _extract_misconceptions_from_attrs(
        self,
        html: str,
        existing: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Wave 81 bridging fallback: harvest misconceptions tagged with
        ``data-cf-misconception="true"`` on the paragraph itself.

        Used when JSON-LD ``misconceptions[]`` is absent or partial. The
        sibling "The right approach" / "Correct approach" subsection (the
        next ``<h4>`` followed by ``<p>`` after the misconception
        paragraph, before the next sibling ``<h4>`` or section close)
        supplies the ``correction`` text. Default ``bloom_level`` is
        ``"analyze"`` per Template 3's typical Bloom range.

        ``existing`` is the list of already-extracted misconceptions
        (from JSON-LD). Any HTML-attr paragraph whose stripped text
        matches an existing entry's ``misconception`` field is skipped
        so JSON-LD wins on duplicates. Dedupe is bidirectional substring
        containment because the HTML-attr paragraph routinely wraps the
        JSON-LD's de-quoted statement with surrounding quotes plus a
        trailing explanatory clause; a strict equality test misses the
        overlap.
        """
        if "data-cf-misconception" not in html:
            return []

        # Build a set of stripped existing misconception statements (case
        # / whitespace / surrounding-quote insensitive) so JSON-LD takes
        # precedence.
        def _norm(s: str) -> str:
            text = re.sub(r"\s+", " ", s or "").strip().lower()
            # Strip ASCII + curly quotes that often wrap the
            # misconception statement on the HTML side.
            text = text.strip('"“”‘’\'')
            return text

        existing_norm: List[str] = []
        for e in existing:
            if not isinstance(e, dict):
                continue
            # Some legacy chunks (pre-Wave-81) store the misconception
            # under ``statement`` rather than the canonical
            # ``misconception``. Honor both to keep dedupe symmetrical
            # across the field-name boundary.
            stmt = e.get("misconception") or e.get("statement")
            if stmt:
                existing_norm.append(_norm(stmt))

        # Find every paragraph carrying data-cf-misconception="true".
        # The regex tolerates additional attributes before or after.
        para_pattern = re.compile(
            r'<p\b[^>]*\bdata-cf-misconception\s*=\s*"true"[^>]*>'
            r'(?P<inner>.*?)</p>',
            re.DOTALL | re.IGNORECASE,
        )

        # "Right approach" / "correct approach" sibling lookup pattern:
        # find an <h4> whose text contains "right approach" or
        # "correct approach", then capture the immediately-following
        # <p>...</p>. Case-insensitive on the heading text.
        correction_pattern = re.compile(
            r'<h4\b[^>]*>\s*(?:[^<]*?)(?:right|correct)\s+approach[^<]*</h4>'
            r'\s*<p\b[^>]*>(?P<correction>.*?)</p>',
            re.DOTALL | re.IGNORECASE,
        )

        added: List[Dict[str, Any]] = []
        for match in para_pattern.finditer(html):
            inner = match.group("inner")
            # Strip nested tags from the misconception paragraph to get
            # plain text. Matches HTMLTextExtractor's behavior loosely.
            misconception_text = re.sub(r"<[^>]+>", "", inner)
            misconception_text = re.sub(
                r"\s+", " ", misconception_text
            ).strip()
            if not misconception_text:
                continue
            norm_candidate = _norm(misconception_text)
            # JSON-LD-wins dedupe with bidirectional substring
            # containment (existing in candidate OR candidate in
            # existing) so quoted-vs-de-quoted variants of the same
            # misconception collapse.
            if any(
                e and (e in norm_candidate or norm_candidate in e)
                for e in existing_norm
            ):
                continue

            entry: Dict[str, Any] = {
                "misconception": misconception_text,
                "bloom_level": "analyze",
            }

            # Look for the correction inside the surrounding section.
            # We bound the search to the enclosing <section>...</section>
            # so paragraphs in unrelated sections don't leak in.
            sec_start = html.rfind("<section", 0, match.start())
            sec_end_marker = html.find("</section>", match.end())
            sec_end = (
                sec_end_marker + len("</section>")
                if sec_end_marker != -1
                else len(html)
            )
            section_slice = html[
                sec_start if sec_start != -1 else 0 : sec_end
            ]
            corr_match = correction_pattern.search(section_slice)
            if corr_match:
                correction_text = re.sub(
                    r"<[^>]+>", "", corr_match.group("correction")
                )
                correction_text = re.sub(
                    r"\s+", " ", correction_text
                ).strip()
                if correction_text:
                    entry["correction"] = correction_text

            # Record this new misconception in the dedupe list so
            # repeated paragraphs (rare but possible) collapse.
            existing_norm.append(norm_candidate)
            added.append(entry)

        return added

    def _extract_title(self, html: str) -> str:
        """Extract page title."""
        # Try <title> tag
        title_match = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()

        # Try <h1>
        h1_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html, re.IGNORECASE)
        if h1_match:
            return h1_match.group(1).strip()

        return "Untitled Module"

    def _extract_sections(self, html: str) -> List[ContentSection]:
        """Extract content sections by heading, including data-cf-* attributes."""
        sections = []

        # Find all headings (capture the full opening tag to read attributes)
        heading_pattern = r'<h([1-6])([^>]*)>([^<]+)</h\1>'
        headings = list(re.finditer(heading_pattern, html, re.IGNORECASE))

        for i, match in enumerate(headings):
            level = int(match.group(1))
            attrs_str = match.group(2)
            heading_text = match.group(3).strip()

            # Get content between this heading and the next
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(html)
            section_html = html[start:end]

            # Extract text
            extractor = HTMLTextExtractor()
            extractor.feed(section_html)
            content = extractor.get_text()

            # Detect components in section
            components = self._detect_components(section_html)

            # Parse data-cf-* attributes from heading tag
            content_type = None
            key_terms: List[str] = []
            ct_match = re.search(r'data-cf-content-type="([^"]*)"', attrs_str)
            if ct_match:
                content_type = ct_match.group(1)
            kt_match = re.search(r'data-cf-key-terms="([^"]*)"', attrs_str)
            if kt_match:
                key_terms = [t.strip() for t in kt_match.group(1).split(",") if t.strip()]

            # REC-VOC-02 (Wave 2, Worker K): scan section body for
            # data-cf-teaching-role attributes on flip-card/self-check/
            # activity components. Courseforge emits these deterministically
            # from (component, purpose) pairs via lib.ontology.teaching_roles.
            tr_matches = re.findall(
                r'data-cf-teaching-role="([^"]*)"', section_html
            )
            distinct_roles = sorted({r for r in tr_matches if r})
            teaching_role = distinct_roles[0] if len(distinct_roles) == 1 else None

            # REC-JSL-03 (Wave 3, Worker M): scan section body for
            # data-cf-objective-ref attributes on .activity-card and
            # .self-check elements. Courseforge emits these from
            # generate_course.py:378,491 when a curriculum JSON entry
            # includes an ``objective_ref``. Deduplicated, deterministic
            # sort so downstream diffs stay stable across runs.
            obj_ref_matches = re.findall(
                r'data-cf-objective-ref="([^"]*)"', section_html
            )
            distinct_obj_refs = sorted({r for r in obj_ref_matches if r})

            # Wave 10: scan section body + heading attrs for
            # ``data-cf-source-ids`` (comma-separated list of DART
            # sourceIds). Courseforge Wave 9 emits these on <section>,
            # headings, and component wrappers (.flip-card, .self-check,
            # .activity-card, .discussion-prompt, .objectives) per the P2
            # scope decision. Each attribute value can list multiple ids
            # separated by commas; split + trim + deduplicate, preserving
            # a sorted order so downstream diffs stay stable.
            source_id_matches: List[str] = []
            for src in re.findall(r'data-cf-source-ids="([^"]*)"', attrs_str):
                source_id_matches.append(src)
            for src in re.findall(r'data-cf-source-ids="([^"]*)"', section_html):
                source_id_matches.append(src)
            distinct_source_ids: List[str] = []
            seen_ids: set = set()
            for raw in source_id_matches:
                for piece in raw.split(","):
                    piece = piece.strip()
                    if piece and piece not in seen_ids:
                        seen_ids.add(piece)
                        distinct_source_ids.append(piece)
            distinct_source_ids.sort()

            sections.append(ContentSection(
                heading=heading_text,
                level=level,
                content=content,
                word_count=len(content.split()),
                components=components,
                content_type=content_type,
                key_terms=key_terms,
                teaching_role=teaching_role,
                teaching_roles=distinct_roles,
                objective_refs=distinct_obj_refs,
                source_references=distinct_source_ids,
            ))

        return sections

    def _extract_objectives(self, html: str,
                             json_ld: Optional[Dict[str, Any]] = None) -> List[LearningObjective]:
        """Extract learning objectives from HTML.

        Priority: JSON-LD > data-cf-* attributes > regex heuristics.
        """
        objectives: List[LearningObjective] = []

        # Strategy 1: JSON-LD (highest fidelity — authoritative Bloom's data)
        if json_ld and json_ld.get("learningObjectives"):
            for lo in json_ld["learningObjectives"]:
                # Wave 69: surface Wave 57 targetedConcepts[] + Wave 59
                # hierarchyLevel/parentObjectiveId so downstream consumers
                # (process_course → build_semantic_graph, inference_rules/
                # targets_concept_from_lo) can materialize the typed LO→
                # concept edges and the terminal/chapter hierarchy tier.
                # Keys translated camelCase (JSON-LD wire format per
                # courseforge_jsonld_v1.schema.json) → snake_case (Trainforge
                # internal convention). Bloom levels lowercased to match
                # Trainforge's case-insensitive ref resolution used by the
                # Wave 66 rule.
                raw_targets = lo.get("targetedConcepts") or []
                targeted: List[Dict[str, str]] = []
                for entry in raw_targets:
                    if not isinstance(entry, dict):
                        continue
                    concept = entry.get("concept")
                    bloom = entry.get("bloomLevel")
                    if not isinstance(concept, str) or not concept:
                        continue
                    if not isinstance(bloom, str) or not bloom:
                        continue
                    targeted.append({
                        "concept": concept,
                        "bloom_level": bloom.lower(),
                    })

                objectives.append(LearningObjective(
                    id=lo.get("id"),
                    text=lo.get("statement", ""),
                    bloom_level=lo.get("bloomLevel"),
                    bloom_verb=lo.get("bloomVerb"),
                    cognitive_domain=lo.get("cognitiveDomain"),
                    key_concepts=lo.get("keyConcepts", []),
                    assessment_suggestions=lo.get("assessmentSuggestions", []),
                    hierarchy_level=lo.get("hierarchyLevel"),
                    parent_objective_id=lo.get("parentObjectiveId"),
                    targeted_concepts=targeted,
                ))
            return objectives

        # Strategy 2: data-cf-* attributes on <li> elements
        cf_li_pattern = re.compile(
            r'<li\s+([^>]*data-cf-objective-id="[^"]*"[^>]*)>(.*?)</li>',
            re.IGNORECASE | re.DOTALL,
        )
        cf_matches = cf_li_pattern.findall(html)
        if cf_matches:
            for attrs_str, inner_html in cf_matches:
                obj_id_m = re.search(r'data-cf-objective-id="([^"]*)"', attrs_str)
                bloom_m = re.search(r'data-cf-bloom-level="([^"]*)"', attrs_str)
                verb_m = re.search(r'data-cf-bloom-verb="([^"]*)"', attrs_str)
                domain_m = re.search(r'data-cf-cognitive-domain="([^"]*)"', attrs_str)
                obj_id = obj_id_m.group(1) if obj_id_m else None
                # Strip HTML tags and objective ID prefix from inner text
                text = re.sub(r'<[^>]+>', '', inner_html).strip()
                text = re.sub(r'^[A-Z]{2,3}-\d+:\s*', '', text).strip()
                bloom_level = bloom_m.group(1) if bloom_m else None
                bloom_verb = verb_m.group(1) if verb_m else None
                domain = domain_m.group(1) if domain_m else None
                if not bloom_level:
                    bloom_level, bloom_verb = self._detect_bloom_level(text)
                objectives.append(LearningObjective(
                    id=obj_id, text=text,
                    bloom_level=bloom_level, bloom_verb=bloom_verb,
                    cognitive_domain=domain,
                ))
            return objectives

        # Strategy 3: Regex fallback (non-Courseforge IMSCC)
        obj_section = re.search(
            r'(?:learning\s+)?objectives?.*?<ul[^>]*>(.*?)</ul>',
            html,
            re.IGNORECASE | re.DOTALL
        )

        if obj_section:
            list_items = re.findall(r'<li[^>]*>([^<]+)</li>', obj_section.group(1))
            for item in list_items:
                text = item.strip()
                bloom_level, bloom_verb = self._detect_bloom_level(text)
                objectives.append(LearningObjective(
                    id=None,
                    text=text,
                    bloom_level=bloom_level,
                    bloom_verb=bloom_verb
                ))

        # Pattern: Structured objective markers (data-objective-id, legacy)
        structured = re.findall(
            r'data-objective-id="([^"]*)"[^>]*>([^<]+)',
            html
        )
        for obj_id, text in structured:
            bloom_level, bloom_verb = self._detect_bloom_level(text)
            objectives.append(LearningObjective(
                id=obj_id,
                text=text.strip(),
                bloom_level=bloom_level,
                bloom_verb=bloom_verb
            ))

        return objectives

    def _detect_bloom_level(self, text: str) -> tuple:
        """Detect Bloom's taxonomy level and verb from objective text.

        Wave 55: delegates to ``lib.ontology.bloom.detect_bloom_level``.
        The pre-Wave-55 local loop used ``startswith() + f" {verb} "`` which
        missed verbs at end-of-text and diverged from the canonical matcher
        on verb-length tie-breaking.
        """
        return _canonical_detect_bloom_level(text)

    CONCEPT_STOP_WORDS = {
        "initial post", "replies", "due", "guidelines", "discussion forum",
        "activity", "question", "feedback", "correct", "incorrect",
        "submit", "deadline", "points", "grading", "rubric",
        "estimated time", "readings", "resources", "learning objectives",
    }

    def _extract_concepts(self, html: str) -> List[str]:
        """Extract key concepts from HTML."""
        concepts = []

        # Look for bold/strong terms
        bold_terms = re.findall(r'<(?:strong|b)[^>]*>([^<]+)</(?:strong|b)>', html)
        concepts.extend([t.strip() for t in bold_terms
                         if len(t.strip()) > 2 and t.strip().lower() not in self.CONCEPT_STOP_WORDS])

        # Look for definition terms
        dt_terms = re.findall(r'<dt[^>]*>([^<]+)</dt>', html)
        concepts.extend([t.strip() for t in dt_terms])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in concepts:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)

        return unique[:20]  # Limit to top 20

    def _detect_components(self, html: str) -> List[str]:
        """Detect interactive components in HTML."""
        components = []

        for component, pattern in self.COMPONENT_PATTERNS.items():
            if re.search(pattern, html, re.IGNORECASE):
                components.append(component)

        return components

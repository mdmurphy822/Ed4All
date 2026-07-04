"""
HTML Content Parser

Extracts structured content from Courseforge-generated HTML modules.
Supports two metadata tiers from Courseforge output:
  1. JSON-LD blocks (<script type="application/ld+json">) — structured page metadata
  2. data-cf-* attributes — inline per-element metadata
Falls back to regex heuristics for non-Courseforge IMSCC packages.
"""

import json as json_mod
import os
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
from lib.ontology.lexical_concept_seeds import is_fragment_phrase as _is_fragment_phrase  # noqa: E402


def _env_flag(name: str) -> bool:
    """Truthy-env-var helper. ``1`` / ``true`` / ``yes`` / ``on`` (case-
    insensitive) are truthy; everything else (incl. unset) is falsey."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


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
    # Wave 81: ``data-cf-template-type`` value harvested from the enclosing
    # ``<section>`` element. Courseforge Wave 79 C content-generator emits this
    # attribute on every section root with values like ``explanation``,
    # ``example``, ``procedure``, ``real_world_scenario``, ``common_pitfall``,
    # ``problem_solution``, ``summary``, ``overview``, ``self_check``. When
    # present, process_course.py prefers this over the heading-keyword heuristic
    # (``_type_from_heading``) so the chunker no longer collapses the four new
    # template types into the legacy six. ``None`` for non-Courseforge IMSCC
    # packages or for legacy Courseforge corpora that predate Wave 79 C.
    template_type: Optional[str] = None
    # Wave 5 (W5.A — ingestion mirror for W1.5): per-block ``keyClaims``
    # array harvested from the JSON-LD ``blocks[]`` projection. Each entry
    # carries at minimum ``claim`` (string) + ``source_chunk_ids``
    # (List[str]) — the chunker uses these to materialize the
    # ``key_claims`` audit field on ``chunk_v4`` so downstream consumers
    # (Trainforge synthesis, NLI claim-support gate) can verify per-claim
    # grounding without re-walking the source HTML. Defensive default
    # ``[]`` so legacy emit paths (pre-W1.5 Courseforge corpora and
    # non-Courseforge IMSCC packages) never see ``None``.
    key_claims: List[Dict[str, Any]] = field(default_factory=list)
    # Wave 5 (W5.A — ingestion mirror for W1.7): per-block
    # ``objectiveAlignment`` array harvested from the JSON-LD ``blocks[]``
    # projection. Each entry declares an ``objective_id`` plus alignment
    # metadata (``status`` / ``declared_bloom`` / etc.) so the
    # ``BlockObjectiveDeliveryValidator`` tri-axis check (NLI / Bloom-gap
    # / verb) has a per-block declaration to validate against rather
    # than re-deriving from the page-level objective list. Defensive
    # default ``[]`` so legacy emit paths never see ``None``.
    objective_alignment: List[Dict[str, Any]] = field(default_factory=list)
    # A7 (end-user-HTML audit, 2026-07-04): the ``data-dart-opener`` role
    # (``objectives`` / ``try_it`` / ``worked_example`` / …) the SemantiK
    # adapter stamps on a promoted pedagogical-opener ``<h4>`` heading. When
    # present it marks a pedagogical sub-unit boundary; the chunker
    # (``chunker._section_is_boundary``) treats it as a soft sub-boundary under
    # ``ED4ALL_CHUNK_SECTION_HARD_BREAK`` so a chunk never fuses an example, its
    # solution, and the next example. ``None`` for non-DART / legacy sections.
    data_dart_opener: Optional[str] = None
    # Wave #22 Tier-2 (composite units): the ``data-dart-unit`` type
    # (``worked_example`` / ``section_opener`` / ``exercise_set`` / …) the
    # SemantiK adapter stamps on the ``<section class="dart-unit">`` wrapper that
    # coagulates a run of sibling blocks into one pedagogical whole. Harvested
    # onto the unit's LEAD section (the wrapper's first child). Marks a preferred
    # chunk boundary — under ``ED4ALL_CHUNK_SECTION_HARD_BREAK`` the chunker
    # (``chunker._section_is_boundary``) breaks at a unit EDGE so a chunk never
    # straddles two composite units. ``None`` for non-unit / legacy sections.
    data_dart_unit: Optional[str] = None
    # Wave #22 quick-wins (chunk pedagogical-role metadata): the distinct
    # ``data-dart-flow`` role values (``statement`` / ``solution-steps`` /
    # ``procedure-steps``) the SemantiK adapter stamps on the BLOCKS inside this
    # section's body (``lib/semantik/adapter.py`` flow-annotation pass /
    # ``_block_attrs``). Harvested from the section body HTML (a section may hold
    # several flow-annotated blocks); sorted + deduped for deterministic
    # downstream diffs. Unioned with ``data_dart_opener`` at chunk-emit time into
    # the additive ``unit_roles`` chunk field. Empty list for non-DART /
    # legacy sections whose body carries no ``data-dart-flow`` attribute.
    data_dart_flows: List[str] = field(default_factory=list)


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
      - Any subtree rooted at an element carrying a ``data-cf-curie``
        attribute. The Courseforge rewrite tier's
        ``RewriteProvider._force_inject_curies`` appends a hidden span —
        ``<span hidden data-cf-curie="ns:concept">ns:concept</span>`` — to
        rewrite HTML when the LLM drops a block's CURIEs. That span is a
        rewrite-tier validator anchor ONLY (so the post-rewrite
        ``rewrite_curie_anchoring`` gate can regex-scrape the tokens); it
        carries synthetic minted CURIE identifiers that exist in no real
        textbook. Skipping its subtree keeps those tokens out of chunk
        ``text`` so the training-synthesis paraphrase pass can't learn to
        emit them. ``data-cf-curie`` is emitted ONLY by force-injection, so
        skipping exactly those elements is precise — legacy / RDF corpora
        and existing fixtures carry no such attribute and extract
        byte-identically.
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
        # Count of currently-open ``data-cf-curie`` ancestors. When nonzero,
        # text data is discarded — mirrors ``_template_chrome_depth``.
        self._curie_anchor_depth = 0
        # Count of currently-open screen-reader-only / a11y-hidden ancestors.
        # When nonzero, text data is discarded — mirrors the two counters
        # above. SemantiK's gold-shell emits screen-reader-only structural
        # labels (``<p class="sr-only" hidden>Paragraph block</p>``) as an
        # accessibility surface; those labels are NOT document content, so
        # left in they leak thousands of "Paragraph block" / "List block"
        # tokens into chunk ``text`` and pollute objectives + retrieval.
        self._a11y_hidden_depth = 0
        # Exemplar-parity wave (A-item pairing): SemantiK now emits real
        # <ul>/<ol>/<table>/<dl> structural bodies (lib/semantik/structure_emit).
        # A flat ``' '.join`` would collapse those back to a run-on string, so
        # the extractor emits lightweight delimiter tokens on the structural
        # boundaries — a newline per <li> / table row / <dd>, a " | " between
        # table cells, and a ": " after each <dt> — so the chunk text keeps the
        # list/table/definition structure signal. Count of cells seen in the
        # current table row (drives the between-cell pipe).
        self._row_cell_count = 0
        # Harvested ``data-cf-curie`` tokens. The subtree text is discarded
        # (376b64f contract) but the CURIE values are kept so the downstream
        # ``curie_anchoring`` gate can still see the force-injected anchors.
        # ``curie_anchors`` is the ordered append log of every space-split
        # token across every ``data-cf-curie`` element seen.
        self.curie_anchors: list[str] = []
        # CURIE tokens whose anchoring element also carried
        # ``data-cf-curie-forced="true"`` (the force-injection marker stamped
        # by ``RewriteProvider._force_inject_curies``). A set — order is not
        # meaningful, and a token may appear on multiple forced spans.
        self.forced_curie_anchors: set[str] = set()

    def _is_template_chrome(self, attrs) -> bool:
        for name, value in attrs:
            if name == "data-cf-role" and value == "template-chrome":
                return True
        return False

    def _is_curie_anchor(self, attrs) -> bool:
        for name, _value in attrs:
            if name == "data-cf-curie":
                return True
        return False

    def _is_a11y_hidden(self, attrs) -> bool:
        """Screen-reader-only / a11y-hidden label element?

        Keyed on a screen-reader-only ``class`` token (``sr-only`` /
        ``visually-hidden`` / …) or ``aria-hidden="true"`` — NOT on a bare
        ``hidden`` attribute. A bare ``hidden`` marks legitimate
        progressive-disclosure reveal content that stays content-bearing (the
        ``data-cf-curie`` force-injection span is skipped by its own signal,
        not by ``hidden``); see ``test_no_curie_attr_identical_to_before``.
        The SemantiK gold-shell labels carry ``class="sr-only"`` so they are
        caught by the class signal regardless of the accompanying ``hidden``.
        """
        for name, value in attrs:
            if name == "class" and value:
                if any(c in _A11Y_HIDDEN_CLASSES for c in value.lower().split()):
                    return True
            elif name == "aria-hidden" and (value or "").strip().lower() == "true":
                return True
        return False

    def _in_skipped_region(self) -> bool:
        """Whether text/delimiters are currently discarded (mirrors handle_data)."""
        return (
            self.in_script
            or self.in_style
            or self._template_chrome_depth > 0
            or self._curie_anchor_depth > 0
            or self._a11y_hidden_depth > 0
        )

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag == 'script':
            self.in_script = True
        elif tag == 'style':
            self.in_style = True
        # A-item pairing — structural delimiters so list/table/dl markup does not
        # collapse to a run-on string. A between-cell " | " on the 2nd+ cell of a
        # row; a row-separating newline on <tr>.
        if tag in ('td', 'th'):
            if self._row_cell_count > 0 and not self._in_skipped_region():
                self.text_parts.append('|')
            self._row_cell_count += 1
        elif tag == 'tr':
            self._row_cell_count = 0
            if not self._in_skipped_region():
                self.text_parts.append('\n')
        if self._is_template_chrome(attrs):
            self._template_chrome_depth += 1
        if self._is_curie_anchor(attrs):
            # Harvest the CURIE tokens BEFORE the subtree-skip swallows the
            # element's text in handle_data. Space-split the attribute value;
            # append each non-empty token to the ordered anchor log. When the
            # element also carries data-cf-curie-forced="true" (the
            # force-injection marker), the same tokens land in the forced set.
            curie_value = ""
            forced_value = ""
            for name, value in attrs:
                if name == "data-cf-curie":
                    curie_value = value or ""
                elif name == "data-cf-curie-forced":
                    forced_value = value or ""
            tokens = [tok for tok in curie_value.split() if tok]
            self.curie_anchors.extend(tokens)
            if forced_value.strip().lower() == "true":
                self.forced_curie_anchors.update(tokens)
            self._curie_anchor_depth += 1
        if self._is_a11y_hidden(attrs):
            self._a11y_hidden_depth += 1

    def handle_endtag(self, tag):
        # A-item pairing — item / definition delimiters (checked BEFORE the
        # chrome/curie/a11y depth decrements below so the skip state is current):
        # a newline per <li> and <dd>, a ": " after each <dt> ("term: definition").
        if tag in ('li', 'dd') and not self._in_skipped_region():
            self.text_parts.append('\n')
        elif tag == 'dt' and not self._in_skipped_region():
            self.text_parts.append(':')
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
        # Close data-cf-curie scope when we see the matching end tag for a
        # curie-anchored element. Same html.parser limitation (no attrs on
        # endtag): force-injection emits the attribute on a ``<span>`` only
        # (see RewriteProvider._force_inject_curies), so the counter
        # decrements on that tag name while inside a curie-anchor region.
        if self._curie_anchor_depth > 0 and tag in _CURIE_ANCHOR_TAGS:
            self._curie_anchor_depth -= 1
        # Close screen-reader-only scope. Same html.parser limitation (no
        # attrs on endtag): SemantiK emits these labels on a leaf ``<p>`` /
        # ``<span>`` carrying only the label text, so the counter decrements
        # on those tag names while inside an a11y-hidden region.
        if self._a11y_hidden_depth > 0 and tag in _A11Y_HIDDEN_TAGS:
            self._a11y_hidden_depth -= 1
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
        # Chrome / curie-anchor self-closers are transient — a self-closing
        # element has no subtree, so the counters are unaffected.

    def handle_data(self, data):
        if self.in_script or self.in_style:
            return
        if self._template_chrome_depth > 0:
            return
        if self._curie_anchor_depth > 0:
            return
        if self._a11y_hidden_depth > 0:
            return
        text = data.strip()
        if text:
            self.text_parts.append(text)

    def get_text(self) -> str:
        return ' '.join(self.text_parts)

    def get_curies(self) -> list[str]:
        """Return every ``data-cf-curie`` token harvested during parsing.

        Ordered append log across all curie-anchored elements. The tokens
        do NOT appear in :meth:`get_text` — the subtree is skipped per the
        376b64f contract — but the downstream ``curie_anchoring`` gate
        consumes this list to verify the force-injected anchors survived.
        """
        return self.curie_anchors

    def get_forced_curies(self) -> list[str]:
        """Return the sorted CURIE tokens carried by force-injected spans.

        A force-injected span is one whose ``data-cf-curie`` element also
        carried ``data-cf-curie-forced="true"``. Sorted for deterministic
        downstream diffs (the backing store is a set).
        """
        return sorted(self.forced_curie_anchors)


# Tags that Courseforge's generate_course.py emits with
# ``data-cf-role="template-chrome"``. Keeping this narrow avoids
# under-counting end tags in complex nested chrome.
_CHROME_TAGS = {"header", "footer", "a", "div", "nav", "aside"}

# Tags that the Courseforge rewrite tier emits with a ``data-cf-curie``
# attribute. ``RewriteProvider._force_inject_curies`` only ever stamps the
# attribute on a ``<span>``, so the end-tag counter only needs to match
# that tag name.
_CURIE_ANCHOR_TAGS = {"span"}

# Class tokens that mark a screen-reader-only / visually-hidden a11y label
# (matched case-insensitively against the space-split ``class`` attribute).
_A11Y_HIDDEN_CLASSES = {
    "sr-only",
    "visually-hidden",
    "visuallyhidden",
    "screen-reader-only",
    "screen-reader-text",
}

# Tags SemantiK's gold-shell emits screen-reader-only labels on. The labels
# are leaf ``<p>`` (block structural labels) / ``<span>`` (inline labels)
# carrying only the label text, so — mirroring ``_CURIE_ANCHOR_TAGS`` — the
# end-tag counter only needs to match those tag names.
_A11Y_HIDDEN_TAGS = {"p", "span"}


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

        # Phase 2 Subtask 30: prefer JSON-LD ``blocks[]`` when present.
        # The Courseforge emitter (gated behind ``COURSEFORGE_EMIT_BLOCKS=true``)
        # publishes a Phase-2 canonical ``blocks[]`` projection of the
        # page's renderer output. When present, project section-typed
        # blocks into ``ContentSection`` entries that complement the
        # legacy regex DOM walk:
        #
        #   - blocks[] wins for metadata fields it carries directly
        #     (content_type / key_terms / template_type).
        #   - The DOM walk wins for structural fields the JSON-LD
        #     doesn't carry (heading text, prose body, level,
        #     components).
        #
        # When ``blocks[]`` is absent (legacy Courseforge corpora,
        # COURSEFORGE_EMIT_BLOCKS=false, or non-Courseforge IMSCC), the
        # block-derived list is empty and ``sections`` keeps the
        # legacy DOM-walk contents unchanged. This preserves the
        # regression contract: Courseforge corpora emit byte-stable
        # under the default flag, and non-Courseforge IMSCC takes the
        # fallback path.
        if json_ld is not None:
            block_entries = self._extract_blocks_from_jsonld(json_ld)
            if block_entries:
                sections.extend(self._content_sections_from_blocks(block_entries))

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
                         "cognitiveDomain", "bloom_level", "cognitive_domain",
                         "cognitiveTaskType", "cognitive_task_type"):
                    continue
                entry[k] = v
            bloom = mc.get("bloomLevel") or mc.get("bloom_level")
            if isinstance(bloom, str) and bloom:
                entry["bloom_level"] = bloom.lower()
            domain = mc.get("cognitiveDomain") or mc.get("cognitive_domain")
            if isinstance(domain, str) and domain:
                entry["cognitive_domain"] = domain
            # GPT Feedback (May 12) item 5: observable cognitive task verb,
            # axis orthogonal to bloom_level. Mirrors the bloomLevel /
            # cognitiveDomain camelCase → snake_case normalization. Optional;
            # silently absent on legacy / pre-Wave-cognitive-task corpora.
            task_type = mc.get("cognitiveTaskType") or mc.get("cognitive_task_type")
            if isinstance(task_type, str) and task_type:
                entry["cognitive_task_type"] = task_type.lower()
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
        # LO-anchoring fix (mirror of the section-level scan): union in
        # ``data-cf-objective-id`` so the page-level fallback set used by
        # the no-sections chunk code path also sees section-root LO ids.
        page_obj_id_matches = re.findall(
            r'data-cf-objective-id="([^"]*)"', html_content
        )
        page_obj_refs = sorted(
            {r for r in (page_obj_ref_matches + page_obj_id_matches) if r}
        )

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

    # Phase 2 (Subtask 30): the subset of canonical Phase-2 ``block_type``
    # values that map onto a ``ContentSection`` heading on the consumer
    # side. The Block dataclass at ``Courseforge/scripts/blocks.py``
    # constrains ``block_type`` to a 16-value enum; the values below are
    # the ones the existing legacy ``_extract_sections`` regex DOM walk
    # also produces sections for. ``content_type_label`` on the Block
    # carries the finer-grained classification (Worker F's 8-value
    # ``SectionContentType`` enum: ``definition`` / ``example`` /
    # ``procedure`` / ``comparison`` / ``exercise`` / ``overview`` /
    # ``summary`` / ``explanation``) and is what the Trainforge consumer
    # ultimately reads as ``ContentSection.content_type``.
    _SECTION_BLOCK_TYPES: frozenset = frozenset({
        "explanation",
        "example",
        "procedure",
        "comparison",
        "definition",
        "overview",
        "summary",
        "exercise",
    })

    def _extract_blocks_from_jsonld(
        self, json_ld: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Phase 2 Subtask 30: surface JSON-LD ``blocks[]`` for direct
        consumption.

        When ``json_ld`` carries a ``blocks`` list (Courseforge emit
        gated behind ``COURSEFORGE_EMIT_BLOCKS=true``), return the list
        directly so callers can map each entry to a ``ContentSection``
        / ``LearningObjective`` without re-walking the DOM. When the
        field is absent (legacy emit, non-Courseforge IMSCC, or
        ``COURSEFORGE_EMIT_BLOCKS=false``), return an empty list and
        the caller falls through to the existing regex DOM walk.

        Wire-key contract (mirrored from Courseforge/scripts/blocks.py
        ``Block.to_jsonld_entry`` and ``schemas/knowledge/courseforge_jsonld_v1.schema.json::$defs/Block``):

          - ``blockId`` — stable position-based ID.
          - ``blockType`` — one of the 16 canonical Phase-2 block types.
          - ``sequence`` — 0/1-indexed position within the page.
          - ``contentTypeLabel`` — finer-grained content classification
            (subset of Worker F's ``SectionContentType`` enum).
          - ``keyTerms`` — slugified key terms attached to the block.
          - ``templateType`` — Courseforge template variant name.

        The shape is camelCase on the wire (JSON-LD convention) and
        the consumer translates to snake_case on the
        ``ContentSection`` / ``LearningObjective`` dataclasses.
        """
        if not isinstance(json_ld, dict):
            return []
        blocks = json_ld.get("blocks")
        if not isinstance(blocks, list):
            return []
        # Filter to dict entries — defensive against malformed payloads
        # that slip past schema validation (e.g. legacy corpora that
        # predate the Wave 67 schema tightening).
        return [b for b in blocks if isinstance(b, dict)]

    def _content_sections_from_blocks(
        self, blocks: List[Dict[str, Any]]
    ) -> List[ContentSection]:
        """Phase 2 Subtask 30: build ``ContentSection`` entries from
        Phase-2 ``blocks[]`` JSON-LD entries.

        Only emits a ``ContentSection`` for blocks whose ``blockType``
        is in :pyattr:`_SECTION_BLOCK_TYPES`. The returned sections
        carry ``content_type`` from ``contentTypeLabel`` (falling back
        to ``blockType`` when the finer-grained label is absent),
        ``key_terms`` from ``keyTerms``, and ``template_type`` from
        ``templateType``. The remaining ``ContentSection`` fields
        (``heading`` / ``level`` / ``content`` / ``word_count`` /
        ``components`` / ``teaching_role`` / ``objective_refs`` /
        ``source_references``) stay empty here — the legacy regex DOM
        walk in :pymeth:`_extract_sections` populates them by walking
        the rendered HTML.

        The caller (``parse``) merges these block-derived sections
        with the DOM-walk sections so the two surfaces complement each
        other: blocks[] wins for the metadata fields it carries
        (``content_type`` / ``key_terms`` / ``template_type``), the
        DOM walk wins for the structural fields the JSON-LD doesn't
        carry (heading text, prose body, level, components).
        """
        sections: List[ContentSection] = []
        for block in blocks:
            block_type = block.get("blockType")
            if block_type not in self._SECTION_BLOCK_TYPES:
                continue
            content_type = block.get("contentTypeLabel") or block_type
            raw_key_terms = block.get("keyTerms") or []
            key_terms: List[str] = [
                kt for kt in raw_key_terms if isinstance(kt, str) and kt
            ]
            template_type = block.get("templateType")
            if not isinstance(template_type, str) or not template_type:
                template_type = None
            # Wave 5 (W5.A): project the W1.5 ``keyClaims`` and W1.7
            # ``objectiveAlignment`` audit arrays. camelCase ↔ snake_case
            # translation: JSON-LD wire keys are camelCase; Python attrs
            # are snake_case. Filter to dict entries — defensive against
            # malformed payloads that slip past schema validation
            # (mirrors the dict-coerce posture in
            # ``_extract_blocks_from_jsonld``).
            raw_key_claims = block.get("keyClaims") or []
            key_claims: List[Dict[str, Any]] = [
                kc for kc in raw_key_claims if isinstance(kc, dict)
            ]
            raw_objective_alignment = block.get("objectiveAlignment") or []
            objective_alignment: List[Dict[str, Any]] = [
                oa for oa in raw_objective_alignment if isinstance(oa, dict)
            ]
            sections.append(
                ContentSection(
                    heading="",
                    level=0,
                    content="",
                    word_count=0,
                    components=[],
                    content_type=content_type,
                    key_terms=key_terms,
                    template_type=template_type,
                    key_claims=key_claims,
                    objective_alignment=objective_alignment,
                )
            )
        return sections

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

    # Wave 81: regex to walk back from a heading to its enclosing
    # ``<section ...>`` open tag and read ``data-cf-template-type``.
    # Courseforge content-generator emits one attribute per section root; the
    # value is the canonical template label that the chunker should honor.
    _SECTION_OPEN_RE = re.compile(
        r'<section\b([^>]*)>', re.IGNORECASE
    )
    _TEMPLATE_TYPE_ATTR_RE = re.compile(
        r'data-cf-template-type="([^"]*)"', re.IGNORECASE
    )
    # LO-anchoring fix: read ``data-cf-objective-id`` off the enclosing
    # ``<section>`` open tag (resolved via the section-walkback that
    # ``_TEMPLATE_TYPE_ATTR_RE`` also uses), not a heading→heading body
    # slice — the body slice crosses into the NEXT section's open tag.
    _OBJECTIVE_ID_ATTR_RE = re.compile(
        r'data-cf-objective-id="([^"]*)"', re.IGNORECASE
    )
    # ``<script>`` / ``<style>`` subtree matcher. Mirrors the canonical
    # stdlib fallback in ``lib/retrieval/citation_anchor.py`` (``re.S | re.I``)
    # so the heading regex below can't see ``<hN>``-shaped fragments inside a
    # JavaScript string literal (e.g. a self-check grading script that builds
    # ``resultEl.innerHTML = '<h3>Your Results</h3>' + feedback``). Such phantom
    # headings would otherwise create fake section boundaries whose body slice
    # is orphaned JS — the slice begins mid-script (no opening ``<script>`` tag),
    # so ``HTMLTextExtractor`` never enters script-skip mode and emits the JS as
    # section ``content``. That JS then can't be located in the script-stripped
    # ``container_text`` at chunk time, so the chunk's ``char_span`` is
    # fabricated (citation anchoring classifies it ``SPAN_FABRICATED`` and the
    # fail-closed citation gate blocks any answer citing it).
    _SCRIPT_STYLE_RE = re.compile(
        r'<(script|style)\b[^>]*>.*?</\1\s*>', re.DOTALL | re.IGNORECASE
    )

    def _extract_sections(self, html: str) -> List[ContentSection]:
        """Extract content sections by heading, including data-cf-* attributes."""
        sections = []

        # Strip ``<script>`` / ``<style>`` subtrees BEFORE the heading regex so
        # ``<hN>``-shaped fragments inside inline JS/CSS string literals can't be
        # mistaken for real section headings (see ``_SCRIPT_STYLE_RE`` above).
        # This method never reads JSON-LD (``_extract_json_ld`` consumes the
        # original ``html_content`` directly), so dropping the
        # ``application/ld+json`` script here is harmless; the data-cf-* /
        # source-id / template-type scans below only read real DOM attributes,
        # which never live inside a script body. Pages without inline script are
        # byte-identical (the regex no-ops).
        html = self._SCRIPT_STYLE_RE.sub(" ", html)

        # Find all headings (capture the full opening tag to read attributes)
        heading_pattern = r'<h([1-6])([^>]*)>([^<]+)</h\1>'
        headings = list(re.finditer(heading_pattern, html, re.IGNORECASE))

        # Wave 81: pre-compute every <section ...> open-tag position so we can
        # walk back from each heading to its nearest enclosing section root and
        # read the data-cf-template-type attribute. Courseforge emits exactly
        # one section root per page (sections do not nest in our content
        # corpus), but the algorithm tolerates nested sections by always
        # taking the closest preceding open tag.
        section_opens = list(self._SECTION_OPEN_RE.finditer(html))

        for i, match in enumerate(headings):
            level = int(match.group(1))
            attrs_str = match.group(2)
            heading_text = match.group(3).strip()

            # Get content between this heading and the next
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(html)
            section_html = html[start:end]

            # Wave 81: derive template_type from the nearest enclosing
            # <section ...> root. Falls back to heading-attr / section-body
            # scan when the section root predates the heading by a wide
            # margin (rare). When no data-cf-template-type is found anywhere,
            # ``template_type`` stays None and process_course.py's heading
            # heuristic continues to drive chunk_type (legacy behavior).
            template_type: Optional[str] = None
            heading_start = match.start()
            for sec_open in reversed(section_opens):
                if sec_open.start() < heading_start:
                    sec_attrs = sec_open.group(1)
                    tt_match = self._TEMPLATE_TYPE_ATTR_RE.search(sec_attrs)
                    if tt_match:
                        template_type = tt_match.group(1).strip() or None
                    break
            # Belt-and-braces: the attribute may also appear directly on the
            # heading or inside the section body (some templates carry it on
            # both the section root and the h1). Pick the first non-empty.
            if not template_type:
                tt_attr = self._TEMPLATE_TYPE_ATTR_RE.search(attrs_str)
                if tt_attr:
                    template_type = tt_attr.group(1).strip() or None
            if not template_type:
                tt_body = self._TEMPLATE_TYPE_ATTR_RE.search(section_html)
                if tt_body:
                    template_type = tt_body.group(1).strip() or None

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

            # A7: harvest the ``data-dart-opener`` role the SemantiK adapter
            # stamps on a promoted pedagogical-opener heading. The adapter puts
            # it on the enclosing ``<section>`` wrapper (which PRECEDES the
            # <h4>), so resolve it the same way ``template_type`` is (Wave 81):
            # walk back to the nearest enclosing ``<section>`` root and read the
            # attribute there; belt-and-braces, also accept it on the heading
            # tag itself. Marks a soft sub-boundary for the hard-break chunker.
            data_dart_opener: Optional[str] = None
            for sec_open in reversed(section_opens):
                if sec_open.start() < heading_start:
                    op_match = re.search(
                        r'data-dart-opener="([^"]*)"', sec_open.group(1)
                    )
                    if op_match and op_match.group(1).strip():
                        data_dart_opener = op_match.group(1).strip()
                    break
            if not data_dart_opener:
                op_attr = re.search(r'data-dart-opener="([^"]*)"', attrs_str)
                if op_attr and op_attr.group(1).strip():
                    data_dart_opener = op_attr.group(1).strip()

            # Wave #22 Tier-2: harvest the ``data-dart-unit`` type off the
            # ``<section class="dart-unit">`` wrapper that encloses this heading's
            # block. The wrapper is the SECTION open immediately PRECEDING the
            # block's own enclosing ``<section>`` (a callout-group ``<div>`` in
            # between is not a section), so a heading is at a unit EDGE exactly
            # when its nearest-enclosing section is the unit wrapper's first
            # child. Marks a preferred chunk boundary.
            data_dart_unit: Optional[str] = None
            nearest_idx: Optional[int] = None
            for j in range(len(section_opens) - 1, -1, -1):
                if section_opens[j].start() < heading_start:
                    nearest_idx = j
                    break
            if nearest_idx is not None:
                nearest_attrs = section_opens[nearest_idx].group(1)
                self_unit = re.search(
                    r'data-dart-unit="([^"]*)"', nearest_attrs
                )
                if self_unit and self_unit.group(1).strip():
                    # The heading's own enclosing section IS a unit wrapper
                    # (rare: a headingless-lead unit whose wrapper is nearest).
                    data_dart_unit = self_unit.group(1).strip()
                elif nearest_idx >= 1:
                    prev_attrs = section_opens[nearest_idx - 1].group(1)
                    prev_unit = re.search(
                        r'data-dart-unit="([^"]*)"', prev_attrs
                    )
                    if prev_unit and prev_unit.group(1).strip():
                        data_dart_unit = prev_unit.group(1).strip()

            # Wave #22 quick-wins: harvest the distinct ``data-dart-flow`` role
            # values off the BLOCKS in this section's body (the SemantiK adapter
            # stamps ``statement`` / ``solution-steps`` / ``procedure-steps`` on
            # worked-example / solution / how-to body blocks). A section body may
            # carry several flow-annotated blocks, so collect every distinct
            # value; sorted for deterministic downstream diffs. Unioned with the
            # section's ``data_dart_opener`` role into the chunk ``unit_roles``
            # metadata at chunk-emit time.
            data_dart_flows = sorted(
                {
                    f.strip()
                    for f in re.findall(
                        r'data-dart-flow="([^"]*)"', section_html
                    )
                    if f.strip()
                }
            )

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
            # LO-anchoring fix: also harvest ``data-cf-objective-id`` — the
            # attribute the Courseforge content-generator stamps on every
            # ``<section>`` root (see e.g. generate_course.py + the page
            # validator ``Courseforge/scripts/validate_page_objectives.py``,
            # which reads ``TO-NN`` LO ids from this exact attribute). The
            # earlier code scanned only ``-ref`` (activity / self-check
            # cards), so content pages whose only LO signal is the section
            # ``-id`` attribute surfaced an empty ``objective_refs`` and the
            # chunker emitted unanchored chunks.
            #
            # The ``-id`` lives on the enclosing ``<section>`` open tag,
            # which precedes the heading — so a body slice (heading→next
            # heading) would cross into the NEXT section's open tag and
            # mis-attribute the id. Resolve it the same way ``template_type``
            # is resolved (Wave 81): walk back to the nearest enclosing
            # ``<section>`` root and read the attribute there. Belt-and-
            # braces: also accept the attribute directly on the heading.
            # Validation against the canonical LO pattern happens downstream
            # in ``extract_learning_outcome_refs`` so free-text values never
            # reach a chunk's ``learning_outcome_refs``.
            obj_id_matches: List[str] = []
            for sec_open in reversed(section_opens):
                if sec_open.start() < heading_start:
                    oid_match = self._OBJECTIVE_ID_ATTR_RE.search(
                        sec_open.group(1)
                    )
                    if oid_match and oid_match.group(1).strip():
                        obj_id_matches.append(oid_match.group(1).strip())
                    break
            heading_oid = self._OBJECTIVE_ID_ATTR_RE.search(attrs_str)
            if heading_oid and heading_oid.group(1).strip():
                obj_id_matches.append(heading_oid.group(1).strip())
            distinct_obj_refs = sorted(
                {r for r in (obj_ref_matches + obj_id_matches) if r}
            )

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
                template_type=template_type,
                data_dart_opener=data_dart_opener,
                data_dart_unit=data_dart_unit,
                data_dart_flows=data_dart_flows,
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
        """Extract key concepts from HTML.

        Bold/strong spans are harvested as candidate concepts. Some source
        notebooks (e.g. NVIDIA's) bold whole SENTENCES rather than key terms,
        which leaks sentence-fragment "concepts" ("Your LLM Can Have",
        "Congratulations We Now Have") into the KG. When
        ``TRAINFORGE_FILTER_FRAGMENT_CONCEPTS`` is set, harvested bold spans
        are passed through the domain-agnostic
        ``lib.ontology.lexical_concept_seeds.is_fragment_phrase`` filter so
        clause-shaped junk is dropped while real noun-phrase concepts
        ("Knowledge Base", "Vector Store") survive. Default off →
        byte-identical legacy bold-harvest, preserving the RDF/SHACL
        calibration corpus.
        """
        concepts = []

        filter_fragments = _env_flag("TRAINFORGE_FILTER_FRAGMENT_CONCEPTS")

        # Look for bold/strong terms
        bold_terms = re.findall(r'<(?:strong|b)[^>]*>([^<]+)</(?:strong|b)>', html)
        for raw in bold_terms:
            term = raw.strip()
            if len(term) <= 2 or term.lower() in self.CONCEPT_STOP_WORDS:
                continue
            if filter_fragments and _is_fragment_phrase(term):
                continue
            concepts.append(term)

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

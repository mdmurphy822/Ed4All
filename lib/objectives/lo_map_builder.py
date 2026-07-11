#!/usr/bin/env python3
"""Build the 3-level Learning Objectives Map blueprint for a course.

The Learning Objectives Map is the course's *blueprint*: not a statements-only
list of objectives, but a navigable 3-level structure that ties every
synthesized learning objective to the instruction blocks that teach it AND to a
deep link into the accessible source textbook the block was grounded against.

  Level 1 — Terminal Objective (``TO-NN``)
  Level 2 — Chapter Objective (``CO-NN``), parented to its TO
  Level 3 — Concept / sub-objective grouping
              └── instruction blocks that target the objective, each carrying
                  its topic (heading text + ``data-cf-content-type``) and a
                  DEEP LINK into the accessible HTML built from the block's
                  ``data-cf-source-ids`` ``dart:<src>#<anchor>`` fragments.

Inputs are passed as ARGUMENTS — never hardcoded:

  * ``objectives`` — the parsed ``synthesized_objectives.json`` doc (either the
    Courseforge ``terminal_objectives`` / ``chapter_objectives`` form or the
    LibV2 archive ``terminal_outcomes`` / ``component_objectives`` form).
  * ``content_dir`` — the course's content-development HTML directory
    (``{project}/03_content_development``), scanned for ``week_*.html`` pages.
  * ``course_code`` / ``slug`` — course identity, threaded into the deep-link
    Studio source route (``/api/learn/source/{slug}?item_path=…#anchor``).

This is a **deterministic** transformation (no LLM, no network). The block →
objective binding is resolved from each block's ``data-cf-objective-id`` /
``data-cf-objective-ref`` attribute; the deep-link target is built from the
block's ``data-cf-source-ids``. Where a CO carries an empty ``sub_objectives``
field (the common case — see finding 17), blocks are grouped under their shared
concept heading as a fallback so the map is always 3 levels deep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

# Single source of truth for parsing the three ``data-dart-pages`` attribute
# forms ("3" / "3-5" / "3,5,7") and the sibling ``data-dart-page-kind`` —
# reused (not re-implemented) from the chunker.
from Trainforge.chunker.helpers import (
    parse_dart_page_kind_attr,
    parse_dart_pages_attr,
)

# Canonical LO id patterns (mirror lib/ontology/learning_objectives.py).
_TO_RE = re.compile(r"^TO-\d{2,}$", re.IGNORECASE)
_CO_RE = re.compile(r"^CO-\d{2,}$", re.IGNORECASE)

# A ``{dart|semantik}:<src>#<anchor>`` provenance fragment (the deep-link
# source). DART->semantik purge Stage 1 (dual-READ): accepts both prefixes.
_DART_REF_RE = re.compile(r"^(?:dart|semantik):(?P<src>[^#]+)#(?P<anchor>.+)$")

# Content-block element discovery. Blocks are ``<section …>`` carrying a
# ``data-cf-block-id`` (+ optionally ``data-cf-source-ids`` /
# ``data-cf-objective-id`` / ``data-cf-content-type``). We parse attributes
# with a tolerant regex rather than a full HTML parser to stay dependency-free
# and to match the deterministic, byte-stable posture of the sibling renderer.
_SECTION_OPEN_RE = re.compile(r"<section\b([^>]*)>", re.IGNORECASE)
_HEADING_RE = re.compile(
    r"<h([1-6])\b([^>]*)>(.*?)</h\1>", re.IGNORECASE | re.DOTALL
)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Embedded non-content elements (JSON-LD metadata, inline CSS/JS). Their bodies
# are NOT instructional prose — their raw text must never become a block topic.
# We strip them from a fragment before deriving any topic summary, and never
# treat a ``<script>`` / ``<style>`` element as a block.
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)

# A leading ``<li …>`` whose text is the objective statement (the Sonnet
# single-pass emitter wraps an objective block's body in an ``<li>`` rather than
# a heading). Its text is a clean, usable topic summary.
_FIRST_LI_RE = re.compile(r"<li\b[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)

# Front-matter / acknowledgments / funder boilerplate that leaks from OpenStax
# front pages (no heading, no instructional value). Conservative: only used to
# DROP a block that ALSO has no heading and no usable label attribute.
_FUNDER_RE = re.compile(
    r"\b(koch foundation|stuart family foundation|"
    r"acknowledg(e?ment)|foundation|donor|grant|copyright|"
    r"all rights reserved|isbn)\b",
    re.IGNORECASE,
)

# A short clean summary keeps at most this many words from a fallback sentence.
_SUMMARY_MAX_WORDS = 12
# Minimum word-character count for a topic to be considered non-garbage.
_MIN_TOPIC_WORD_CHARS = 3


def _attr(attrs: str, name: str) -> Optional[str]:
    """Extract one HTML attribute value from a raw attribute string."""
    m = re.search(
        rf'{re.escape(name)}\s*=\s*"([^"]*)"', attrs, re.IGNORECASE
    )
    if m:
        return m.group(1)
    m = re.search(
        rf"{re.escape(name)}\s*=\s*'([^']*)'", attrs, re.IGNORECASE
    )
    return m.group(1) if m else None


def _text_of(fragment: str) -> str:
    """Strip tags + collapse whitespace from an HTML fragment to plain text.

    Embedded ``<script>`` (incl. JSON-LD) and ``<style>`` bodies are removed
    first so their raw payload never bleeds into a derived topic.
    """
    fragment = _SCRIPT_STYLE_RE.sub(" ", fragment)
    text = _TAG_STRIP_RE.sub(" ", fragment)
    text = unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _word_char_count(text: str) -> int:
    """Count alphabetic word characters (drives the garbage / drop heuristic)."""
    return sum(1 for ch in text if ch.isalpha())


def _is_number_dominated(text: str, threshold: float = 0.6) -> bool:
    """True when more than ``threshold`` of the tokens are pure numbers.

    Used (with a no-heading guard) to drop number-only chunks like
    ``"37889005 16 62008465"`` / ``"84 90 69 55 88 70 60 72"`` that the emitter
    sometimes scraped into a block-id slug or first text node.
    """
    tokens = [t for t in text.split() if t]
    if not tokens:
        return True
    numeric = sum(
        1 for t in tokens if re.fullmatch(r"[\d.,%/+\-x×–]+", t)
    )
    return numeric / len(tokens) > threshold


def _is_garbage_topic(text: str) -> bool:
    """True when ``text`` is too short / number-dominated to be a real topic."""
    text = text.strip()
    if not text:
        return True
    if _word_char_count(text) < _MIN_TOPIC_WORD_CHARS:
        return True
    return _is_number_dominated(text)


def _clean_summary(text: str) -> str:
    """Derive a SHORT clean topic summary from a block's plain text.

    Takes the first sentence (or clause), drops leading pure-number runs, and
    truncates to ``_SUMMARY_MAX_WORDS`` words. Returns ``""`` when nothing
    usable remains (the caller then drops the block).
    """
    text = (text or "").strip()
    if not text:
        return ""
    # First sentence / clause.
    first = re.split(r"(?<=[.!?])\s+|[—;:]", text, maxsplit=1)[0].strip()
    if not first:
        first = text
    # Drop leading pure-number / punctuation tokens (scraped figure labels).
    tokens = first.split()
    while tokens and re.fullmatch(r"[\d.,%/+\-x×–()]+", tokens[0]):
        tokens.pop(0)
    summary = " ".join(tokens[:_SUMMARY_MAX_WORDS]).strip()
    if _is_garbage_topic(summary):
        return ""
    return summary


# --------------------------------------------------------------------------- #
# Deep-link construction.
# --------------------------------------------------------------------------- #


def parse_dart_ref(ref: str) -> Optional[Tuple[str, str]]:
    """Split a ``dart:<src>#<anchor>`` provenance id into ``(src, anchor)``.

    Returns ``None`` for any non-``dart:`` / malformed ref. ``src`` is the
    accessible-source basename (e.g. ``sample-book-ch1-3_accessible``)
    and ``anchor`` is the in-page ``id=`` fragment.
    """
    if not isinstance(ref, str):
        return None
    m = _DART_REF_RE.match(ref.strip())
    if not m:
        return None
    src = m.group("src").strip()
    anchor = m.group("anchor").strip()
    if not src or not anchor:
        return None
    return src, anchor


def deep_link_for_ref(
    ref: str, slug: str, page: Optional[int] = None
) -> Optional[str]:
    """Build a Studio source-route deep link for one ``dart:<src>#<anchor>`` ref.

    Mirrors ``gui/services/answer_render.source_url_for``: the canonical viewer
    route is ``/api/learn/source/{slug}?item_path={src}.html#{anchor}``. The
    ``src`` basename maps to the archived ``source/html/{src}.html`` file the
    citation links into; the anchor is the in-page ``id=`` the leak-sanitizer
    strips from rendered content but which must be SURFACED here as a link
    (finding 14). Returns ``None`` for a non-resolvable ref.

    When ``page`` is a positive int (parsed from the block's ``data-dart-pages``
    attribute via ``parse_dart_pages_attr``), ``&page=N`` is appended BEFORE the
    ``#anchor`` fragment so the serve-time ``id="page-N"`` anchor resolves
    (page-number deep-links, Phase 1). The page-LESS path
    (``page is None``) is byte-identical to today — no ``&page=`` param.
    Anti-fabrication: callers pass only a page that is actually present on
    ``data-dart-pages`` (never an invented one).
    """
    parsed = parse_dart_ref(ref)
    if parsed is None:
        return None
    src, anchor = parsed
    item_path = f"{src}.html"
    url = "/api/learn/source/{slug}?item_path={item}".format(
        slug=quote(str(slug), safe=""),
        item=quote(item_path, safe=""),
    )
    if isinstance(page, int) and page > 0:
        url = "{url}&page={page}".format(url=url, page=page)
    return "{url}#{frag}".format(url=url, frag=quote(anchor, safe="-"))


def _first_source_ids(attrs: str) -> List[str]:
    """Parse a block's ``data-cf-source-ids`` CSV into a ref list."""
    raw = _attr(attrs, "data-cf-source-ids") or ""
    return [r.strip() for r in raw.split(",") if r.strip()]


# --------------------------------------------------------------------------- #
# Data model.
# --------------------------------------------------------------------------- #


@dataclass
class InstructionBlock:
    """One instruction block bound to an objective, with topic + deep link."""

    block_id: str
    content_type: str  # data-cf-content-type, or inferred from block_id
    topic: str  # heading text (the block's subject)
    objective_id: str  # data-cf-objective-id / -ref the block targets
    page_id: str  # owning page (week_NN_*)
    deep_link: Optional[str]  # first resolvable dart source deep link
    source_refs: List[str] = field(default_factory=list)  # all dart refs
    pages: List[int] = field(default_factory=list)  # data-dart-pages
    # data-dart-page-kind: "printed" | "interpolated" | "physical". Absent →
    # "physical" (back-compat — the whole existing corpus has no kind attr).
    # Drives the kind-aware "p. N" (printed) vs "PDF p. N" (physical) label.
    pages_kind: str = "physical"

    def block_type_label(self) -> str:
        """Human-readable single block-type label (no duplication, no filler)."""
        return (self.content_type or "content").replace("_", " ").strip()

    def page_citation(self) -> str:
        """Kind-aware page citation for this block's link surface.

        Delegates to ``lib.page_label.page_citation`` with the block's DART-
        asserted ``pages_kind``: ``"p. 47"`` for a PRINTED / ``interpolated``
        page, ``"PDF p. 47"`` for a ``physical`` page (or an absent kind, which
        is normalized to ``physical`` — so a kind-less corpus is byte-identical
        to the legacy ``pdf_page_citation`` output). ``""`` when page-less.
        """
        from lib.page_label import page_citation  # noqa: PLC0415

        pages = [p for p in (self.pages or []) if isinstance(p, int)]
        return page_citation(pages, self.pages_kind)

    def label(self) -> str:
        """Concise map label: ``"{topic} — {block_type}"``.

        The TO/CO parent already supplies the objective context, so the label
        carries only the block's own topic + its content type — no duplicated
        type, no ``"that develops TO-NN"`` filler.
        """
        topic = (self.topic or "").strip()
        bt = self.block_type_label()
        if topic and bt:
            return f"{topic} — {bt}"
        return topic or bt


@dataclass
class ConceptGroup:
    """Level-3 grouping: a concept / sub-objective + its instruction blocks."""

    label: str
    blocks: List[InstructionBlock] = field(default_factory=list)


@dataclass
class ObjectiveNode:
    """A chapter objective (CO) with its concept groups (level 2 → 3)."""

    id: str
    statement: str
    bloom_level: str = ""
    terminal_id: Optional[str] = None
    concepts: List[ConceptGroup] = field(default_factory=list)

    def block_count(self) -> int:
        return sum(len(c.blocks) for c in self.concepts)


@dataclass
class TerminalNode:
    """A terminal objective (TO) with its child COs (level 1 → 2)."""

    id: str
    statement: str
    bloom_level: str = ""
    children: List[ObjectiveNode] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Objective normalization (shared shape with the renderer).
# --------------------------------------------------------------------------- #


def _norm_objectives(
    doc: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Normalize either objective shape to ``(terminals, components)``.

    Accepts the Courseforge form (``terminal_objectives`` +
    ``chapter_objectives`` as a dict-by-chapter, a list-of-groups
    ``[{"chapter", "objectives": [...]}]``, or a flat CO list) and the LibV2
    archive form (``terminal_outcomes`` + ``component_objectives`` with
    ``parent_terminal``). Identical contract to
    ``render_learning_objectives_page._norm_objectives``.
    """
    terminals_raw = (
        doc.get("terminal_objectives") or doc.get("terminal_outcomes") or []
    )
    terminals = [t for t in terminals_raw if isinstance(t, dict)]

    components: List[Dict[str, Any]] = []
    chapter_obj = doc.get("chapter_objectives")
    if isinstance(chapter_obj, dict):
        for chapter_key in chapter_obj:
            bucket = chapter_obj[chapter_key]
            if not isinstance(bucket, list):
                continue
            for co in bucket:
                if isinstance(co, dict):
                    co = dict(co)
                    co.setdefault("chapter", chapter_key)
                    components.append(co)
    elif isinstance(chapter_obj, list):
        is_grouped = any(
            isinstance(el, dict) and isinstance(el.get("objectives"), list)
            for el in chapter_obj
        )
        if is_grouped:
            for grp in chapter_obj:
                if not isinstance(grp, dict):
                    continue
                inner = grp.get("objectives")
                if not isinstance(inner, list):
                    continue
                for co in inner:
                    if isinstance(co, dict):
                        co = dict(co)
                        co.setdefault("chapter", grp.get("chapter"))
                        components.append(co)
        else:
            components = [c for c in chapter_obj if isinstance(c, dict)]
    else:
        archive = doc.get("component_objectives")
        if isinstance(archive, list):
            for co in archive:
                if isinstance(co, dict):
                    co = dict(co)
                    if "terminal_id" not in co and co.get("parent_terminal"):
                        co["terminal_id"] = co["parent_terminal"]
                    components.append(co)

    return terminals, components


def _co_parent(co: Dict[str, Any]) -> Optional[str]:
    """Resolve a CO's parent terminal id (upper-cased), or ``None``."""
    pid = co.get("terminal_id") or co.get("parent_terminal")
    if isinstance(pid, str) and _TO_RE.match(pid.strip()):
        return pid.strip().upper()
    return None


def _co_sub_objectives(co: Dict[str, Any]) -> List[str]:
    """Extract a CO's declared sub-objective labels (finding 17), or ``[]``."""
    subs = co.get("sub_objectives") or []
    out: List[str] = []
    for s in subs:
        if isinstance(s, str) and s.strip():
            out.append(s.strip())
        elif isinstance(s, dict):
            label = s.get("statement") or s.get("label") or s.get("id")
            if label and str(label).strip():
                out.append(str(label).strip())
    return out


# --------------------------------------------------------------------------- #
# Content-block extraction.
# --------------------------------------------------------------------------- #


def _content_type_from_block_id(block_id: str) -> str:
    """Infer a content type from a ``page#blocktype_slug_N`` block id."""
    if "#" not in block_id:
        return "content"
    tail = block_id.split("#", 1)[1]
    # tail is ``blocktype_concept-slug_N`` — the leading token is the type.
    return tail.split("_", 1)[0] if "_" in tail else tail


def _concept_label_from_block_id(block_id: str) -> str:
    """Derive a human concept label from a ``page#blocktype_slug_N`` id."""
    if "#" not in block_id:
        return ""
    tail = block_id.split("#", 1)[1]
    parts = tail.split("_")
    # Drop the leading blocktype token and a trailing numeric sequence token.
    if len(parts) >= 2:
        middle = parts[1:-1] if parts[-1].isdigit() else parts[1:]
    else:
        middle = parts
    slug = "_".join(middle)
    return slug.replace("-", " ").replace("_", " ").strip()


def extract_blocks_from_html(
    html: str, page_id: str, slug: str
) -> List[InstructionBlock]:
    """Extract instruction blocks bound to an objective from one HTML page.

    A block is a ``<section>`` carrying BOTH a ``data-cf-block-id`` and a
    ``data-cf-objective-id`` / ``data-cf-objective-ref``. The block's *topic*
    is the text of the first heading inside it (or its block-id-derived concept
    slug when no heading is present); its *deep link* is the first resolvable
    ``dart:<src>#<anchor>`` from its ``data-cf-source-ids``.

    A page wraps each block in an outer ``<section data-cf-block-id …>`` and an
    inner ``<section data-cf-source-ids … data-cf-content-type …>`` carrying the
    heading. We dedupe on ``block_id`` (first occurrence wins) and enrich the
    record from any later section sharing that id.
    """
    matches = list(_SECTION_OPEN_RE.finditer(html))
    by_id: Dict[str, InstructionBlock] = {}
    order: List[str] = []
    _explicit_ctype: Dict[str, bool] = {}
    # Per-block raw plain text (heading-stripped) for the summary fallback +
    # the non-instructional drop heuristic. Keyed by block_id.
    _block_text: Dict[str, str] = {}
    _has_heading: Dict[str, bool] = {}
    _label_attr: Dict[str, str] = {}

    for idx, m in enumerate(matches):
        attrs = m.group(1)
        block_id = _attr(attrs, "data-cf-block-id") or ""
        objective_id = (
            _attr(attrs, "data-cf-objective-id")
            or _attr(attrs, "data-cf-objective-ref")
            or ""
        ).strip()
        content_type = (_attr(attrs, "data-cf-content-type") or "").strip()
        source_ids = _first_source_ids(attrs)
        # Physical source pages off the SAME element the chunker harvests from
        # (``data-dart-pages``); reuse the canonical chunker parser for the
        # "3" / "3-5" / "3,5,7" forms. Empty / absent → [] (page-less path).
        block_pages = parse_dart_pages_attr(_attr(attrs, "data-dart-pages"))
        # Sibling ``data-dart-page-kind`` on the SAME element — DART asserts
        # whether the pages are the book's PRINTED page or the PDF/physical
        # page. Absent → "physical" (back-compat). Drives the honest label.
        block_pages_kind = parse_dart_page_kind_attr(
            _attr(attrs, "data-dart-page-kind")
        )

        if not block_id:
            continue

        # Section body (bounded to before the next section open). Strip embedded
        # <script> (incl. JSON-LD) / <style> so their payload never leaks.
        next_start = (
            matches[idx + 1].start() if idx + 1 < len(matches) else len(html)
        )
        raw_inner = html[m.end():next_start]
        inner = _SCRIPT_STYLE_RE.sub(" ", raw_inner)

        heading_text = ""
        hm = _HEADING_RE.search(inner)
        if hm:
            heading_text = _text_of(hm.group(3))
            if not content_type:
                content_type = (_attr(hm.group(2), "data-cf-content-type") or "").strip()

        # A data-cf label attribute is the next-best topic when no heading.
        label_attr = (
            _attr(attrs, "data-cf-term")
            or _attr(attrs, "data-cf-key-terms")
            or ""
        ).strip()
        # First key-term only (key-terms is a CSV).
        if label_attr and "," in label_attr:
            label_attr = label_attr.split(",", 1)[0].strip()

        # Plain text of the block with its heading removed (summary source).
        body_no_heading = _HEADING_RE.sub(" ", inner)
        body_text = _text_of(body_no_heading)

        if heading_text and _word_char_count(heading_text) >= _MIN_TOPIC_WORD_CHARS:
            _has_heading[block_id] = True
        _block_text.setdefault(block_id, "")
        if body_text and not _block_text.get(block_id):
            _block_text[block_id] = body_text
        if label_attr and not _label_attr.get(block_id):
            _label_attr[block_id] = label_attr

        existing = by_id.get(block_id)
        if existing is None:
            existing = InstructionBlock(
                block_id=block_id,
                content_type=content_type
                or _content_type_from_block_id(block_id),
                topic=heading_text,
                objective_id=objective_id,
                page_id=page_id,
                deep_link=None,
                source_refs=[],
                pages=list(block_pages),
                pages_kind=block_pages_kind,
            )
            # Track whether the type came from an explicit data-cf-content-type
            # attribute (authoritative) or a block-id inference (a guess that an
            # explicit attribute on a sibling/inner section should override).
            _explicit_ctype[block_id] = bool(content_type)
            by_id[block_id] = existing
            order.append(block_id)
        else:
            # Enrich the first record from a sibling/inner section.
            if not existing.objective_id and objective_id:
                existing.objective_id = objective_id
            # An explicit content-type on any section wins over a block-id guess.
            if content_type and not _explicit_ctype.get(block_id):
                existing.content_type = content_type
                _explicit_ctype[block_id] = True
            if not existing.topic and heading_text:
                existing.topic = heading_text
            # Enrich pages from a sibling/inner section carrying them. The kind
            # rides with the pages (the same element asserts both), so adopt the
            # sibling's kind only when adopting its pages.
            if not existing.pages and block_pages:
                existing.pages = list(block_pages)
                existing.pages_kind = block_pages_kind

        # Merge source refs + resolve a deep link from the first usable one.
        for ref in source_ids:
            if ref not in existing.source_refs:
                existing.source_refs.append(ref)
        if existing.deep_link is None:
            # Deep-link to the FIRST physical page when known (RISK-C: a
            # "3-5" range targets page 3); page-less blocks keep the byte-
            # identical URL with no ``&page=`` param.
            first_page = existing.pages[0] if existing.pages else None
            for ref in existing.source_refs:
                link = deep_link_for_ref(ref, slug, page=first_page)
                if link:
                    existing.deep_link = link
                    break

    # Resolve each block's topic (heading > label attr > clean summary) and
    # drop blocks that are non-instructional chrome / front-matter or that have
    # no usable topic at all (raw number runs, funder boilerplate).
    kept: List[InstructionBlock] = []
    for bid in order:
        blk = by_id[bid]
        has_heading = _has_heading.get(bid, False)
        body_text = _block_text.get(bid, "")
        label_attr = _label_attr.get(bid, "")

        # (3) Drop clearly non-instructional blocks: NO heading AND
        # (number-dominated OR funder/acknowledgment boilerplate). Conservative
        # — a real heading always keeps the block.
        if not has_heading and not label_attr:
            if _FUNDER_RE.search(body_text) or (
                body_text and _is_number_dominated(body_text)
            ):
                continue

        # (2) Topic from heading > data-cf label attr > clean summary.
        topic = (blk.topic or "").strip()
        if not (topic and not _is_garbage_topic(topic)):
            topic = ""
        if not topic and label_attr and not _is_garbage_topic(label_attr):
            topic = label_attr
        if not topic:
            topic = _clean_summary(body_text)

        # No heading, no label, no usable summary → drop rather than emit noise.
        if not topic or _is_garbage_topic(topic):
            continue

        blk.topic = topic
        kept.append(blk)

    # Only blocks that actually bind to an objective belong on the map.
    return [blk for blk in kept if blk.objective_id]


def load_blocks_from_content_dir(
    content_dir: Path, slug: str
) -> List[InstructionBlock]:
    """Scan a content-development dir for ``week_*.html`` pages → blocks.

    ``content_dir`` is passed by the caller (never hardcoded). Non-week pages
    (course_overview, manifests) are skipped. Pages are read in sorted order
    for deterministic output.
    """
    blocks: List[InstructionBlock] = []
    if not content_dir.is_dir():
        return blocks
    # Support both the flat layout (``week_01_content_01.html`` at the top
    # level) and the nested layout (``week_01/content_01.html``). The page_id
    # is normalised so a nested ``week_01/content_01`` reads as
    # ``week_01_content_01`` (matching the flat naming downstream).
    flat = sorted(content_dir.glob("week_*.html"))
    nested = sorted(
        p for p in content_dir.glob("week_*/*.html") if p.is_file()
    )
    for page in flat + nested:
        if page.parent == content_dir:
            page_id = page.stem
        else:
            page_id = "{}_{}".format(page.parent.name, page.stem)
        try:
            html = page.read_text(encoding="utf-8")
        except OSError:
            continue
        blocks.extend(extract_blocks_from_html(html, page_id, slug))
    return blocks


# --------------------------------------------------------------------------- #
# Blueprint assembly.
# --------------------------------------------------------------------------- #


# Stop-words excluded from concept-label token-overlap matching (low-signal
# filler that would inflate spurious overlaps between a block heading and a
# declared sub-objective label).
_LABEL_STOPWORDS = frozenset(
    {
        "a", "an", "and", "the", "of", "to", "for", "in", "on", "by", "with",
        "is", "are", "as", "at", "or", "be", "using", "use", "your", "you",
    }
)


def _label_tokens(text: str) -> set:
    """Lower-cased alphabetic content tokens of a label (stop-words dropped)."""
    toks = re.findall(r"[a-z]+", (text or "").lower())
    return {t for t in toks if len(t) > 1 and t not in _LABEL_STOPWORDS}


def _group_blocks_into_concepts(
    co: Dict[str, Any], blocks: List[InstructionBlock]
) -> List[ConceptGroup]:
    """Level-3 grouping for one CO.

    Uses the CO's declared ``sub_objectives`` as concept labels when present
    (finding 17); otherwise groups blocks by their shared concept heading
    (the block's cleaned ``topic`` text) as a fallback so the map is always 3
    levels deep.

    Concept-group labelling: the block's already-resolved, de-garbaged
    ``topic`` (the real heading text that ``extract_blocks_from_html`` cleans)
    is the authoritative concept label. The block-id-derived slug
    (``_concept_label_from_block_id``) is only a LAST-RESORT fallback for a
    block whose ``topic`` is empty / garbage — it yields truncated fragments
    ("place value in", "what you will") so it must never override a real
    heading.
    """
    declared = _co_sub_objectives(co)
    if declared:
        # One concept group per declared sub-objective. When blocks carry
        # distinct headings, attach each block under the declared concept it
        # best matches by token overlap; an unmatched block (or a block whose
        # heading is absent) falls under the first declared concept so no block
        # is ever dropped. Each declared concept stays visible as its own
        # level-3 node (possibly empty) so the nesting is honest.
        groups = [ConceptGroup(label=lbl) for lbl in declared]
        if not groups:
            return groups
        declared_tokens = [_label_tokens(lbl) for lbl in declared]
        for blk in blocks:
            blk_tokens = _label_tokens(blk.topic or "")
            best_idx = 0
            best_overlap = 0
            if blk_tokens:
                for i, dt in enumerate(declared_tokens):
                    overlap = len(blk_tokens & dt)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = i
            groups[best_idx].blocks.append(blk)
        return groups

    # Fallback: cluster by concept label. Prefer the block's CLEANED topic
    # (the real heading text) over the block-id slug, which is truncated and can
    # carry the same front-matter noise the topic-resolver already scrubbed.
    # Blocks sharing the same cleaned concept label are grouped together.
    by_concept: Dict[str, ConceptGroup] = {}
    order: List[str] = []
    for blk in blocks:
        topic = (blk.topic or "").strip()
        if topic and not _is_garbage_topic(topic):
            label = topic
        else:
            slug = _concept_label_from_block_id(blk.block_id)
            if slug and not _is_garbage_topic(slug) and not _FUNDER_RE.search(slug):
                label = slug
            else:
                label = "Core concepts"
        key = label.lower()
        grp = by_concept.get(key)
        if grp is None:
            grp = ConceptGroup(label=label)
            by_concept[key] = grp
            order.append(key)
        grp.blocks.append(blk)
    return [by_concept[k] for k in order]


def build_lo_map(
    objectives: Dict[str, Any],
    content_dir: Path,
    *,
    slug: str,
) -> Tuple[List[TerminalNode], List[ObjectiveNode]]:
    """Assemble the 3-level Learning Objectives Map.

    Returns ``(terminals, unmapped_cos)`` where ``terminals`` is the ordered
    list of ``TerminalNode`` (TO → CO → concept → block) and ``unmapped_cos``
    holds COs with no resolvable parent TO (rendered under an honest
    "Unmapped" grouping, never fabricated under an arbitrary TO).

    Block → objective binding (non-additive): each instruction block declares
    the objective it targets via ``data-cf-objective-id`` / ``-ref``. A CO's
    blocks are the blocks stamped with EXACTLY that CO id. Only when a CO has
    ZERO directly-stamped blocks does it FALL BACK to the parent TO's
    directly-stamped blocks (so a CO never renders empty when only TO-level
    stamps exist — preserving graceful degradation for the legacy emitter that
    stamps the TO id on content blocks, finding 14). The TO-stamped pile is NOT
    added on top of a CO's own blocks, so once the upstream data stamps specific
    CO ids each CO shows its OWN blocks instead of identical fan-out piles.
    """
    terminals, components = _norm_objectives(objectives)
    blocks = load_blocks_from_content_dir(content_dir, slug)

    # Index blocks by the objective id they declare.
    blocks_by_obj: Dict[str, List[InstructionBlock]] = {}
    for blk in blocks:
        blocks_by_obj.setdefault(blk.objective_id.upper(), []).append(blk)

    # Build CO nodes, grouped by parent TO.
    co_nodes_by_parent: Dict[str, List[ObjectiveNode]] = {}
    unmapped: List[ObjectiveNode] = []
    for co in components:
        cid = str(co.get("id", "")).strip().upper()
        if not cid:
            continue
        parent = _co_parent(co)
        # Blocks that bind directly to this CO. Only when a CO has NO directly-
        # stamped blocks of its own does it fall back to the parent TO's
        # directly-stamped blocks (graceful degradation for TO-only stamps) —
        # the TO pile is a FALLBACK, never added on top of direct blocks.
        co_blocks = list(blocks_by_obj.get(cid, []))
        if not co_blocks and parent:
            co_blocks = list(blocks_by_obj.get(parent, []))
        # Dedupe by block_id (defensive — never duplicate within one CO).
        seen: set = set()
        deduped: List[InstructionBlock] = []
        for b in co_blocks:
            if b.block_id not in seen:
                seen.add(b.block_id)
                deduped.append(b)

        node = ObjectiveNode(
            id=cid,
            statement=str(co.get("statement", "")),
            bloom_level=str(co.get("bloom_level", "")).strip().lower(),
            terminal_id=parent,
            concepts=_group_blocks_into_concepts(co, deduped),
        )
        if parent:
            co_nodes_by_parent.setdefault(parent, []).append(node)
        else:
            unmapped.append(node)

    terminal_nodes: List[TerminalNode] = []
    for t in terminals:
        tid = str(t.get("id", "")).strip().upper()
        if not tid:
            continue
        children = co_nodes_by_parent.get(tid, [])

        # Graceful degradation: when no CO children are mapped to this TO but
        # instruction blocks are stamped directly with the TO id (the Sonnet /
        # single-pass emitter stamps TO ids on blocks when no CO layer exists),
        # synthesize a thin proxy ObjectiveNode that surfaces those blocks.
        # This makes the map useful for TO-only courses instead of showing each
        # TO as "No chapter objectives are mapped."
        if not children:
            to_blocks = blocks_by_obj.get(tid, [])
            if to_blocks:
                # Honest TO-direct grouping: these blocks teach the TO directly
                # (this course has no chapter-objective layer), so do NOT mint a
                # fake CO that duplicates the TO statement. Use a neutral label
                # and an empty id so the renderer shows it as a block grouping,
                # not a phantom chapter objective.
                proxy = ObjectiveNode(
                    id="",
                    statement="Instruction blocks for this objective",
                    bloom_level="",
                    terminal_id=None,  # self-referential proxy; no separate CO
                    concepts=_group_blocks_into_concepts(t, to_blocks),
                )
                children = [proxy]

        terminal_nodes.append(
            TerminalNode(
                id=tid,
                statement=str(t.get("statement", "")),
                bloom_level=str(t.get("bloom_level", "")).strip().lower(),
                children=children,
            )
        )

    return terminal_nodes, unmapped


# --------------------------------------------------------------------------- #
# Per-TO summary aggregation (drives the map page's top summary table).
# --------------------------------------------------------------------------- #


@dataclass
class TerminalSummaryRow:
    """One summary-table row: per-TO rolled-up counts.

    Pure aggregation over a ``TerminalNode`` — no new external data, so the
    same numbers always match the detailed outline rendered below the table.
    ``num_cos`` counts only REAL chapter-objective children (the TO-only proxy
    grouping, whose ``ObjectiveNode.id`` is empty, is excluded so a TO-only
    course honestly shows 0 COs). ``num_blocks`` is the deduped instruction-
    block count across the TO's children; ``num_source_links`` is the count of
    DISTINCT resolvable deep links those blocks carry (the deep-link targets a
    learner can follow into the accessible source textbook).
    """

    terminal_id: str
    statement: str
    bloom_level: str
    num_cos: int
    num_blocks: int
    num_source_links: int


def summarize_terminals(tos: List[TerminalNode]) -> List[TerminalSummaryRow]:
    """Roll a list of ``TerminalNode`` up to one summary row per TO.

    Pure function (no I/O, no LLM) consumed by the map renderer's summary
    table. Generic over any course: the proxy TO-direct grouping (empty-id
    child synthesized for TO-only courses) is excluded from ``num_cos`` so such
    a course renders ``0`` COs gracefully, while its instruction blocks + source
    links are still counted. Blocks + source links are deduped (by ``block_id``
    / by deep-link string) so a TO-stamped block fanned out across several COs
    is counted once per TO.
    """
    rows: List[TerminalSummaryRow] = []
    for to in tos:
        real_cos = sum(1 for co in to.children if (co.id or "").strip())
        seen_blocks: set = set()
        seen_links: set = set()
        for co in to.children:
            for concept in co.concepts:
                for blk in concept.blocks:
                    if blk.block_id in seen_blocks:
                        continue
                    seen_blocks.add(blk.block_id)
                    if blk.deep_link:
                        seen_links.add(blk.deep_link)
        rows.append(
            TerminalSummaryRow(
                terminal_id=to.id,
                statement=to.statement,
                bloom_level=to.bloom_level,
                num_cos=real_cos,
                num_blocks=len(seen_blocks),
                num_source_links=len(seen_links),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Reusable objective-list renderer (finding 18 — overview page imports this).
# --------------------------------------------------------------------------- #


def render_objective_list(
    tos: List[Dict[str, Any]],
    cos: List[Dict[str, Any]],
    *,
    heading_level: int = 3,
) -> str:
    """Render a TO + each of its COs (with statements) as a semantic list.

    The overview page (finding 18 — "overview must list TO + all its COs + each
    statement") imports this so a week's overview enumerates each chapter
    objective with its statement as a brief explanation, instead of listing the
    week's TO alone.

    ``tos`` / ``cos`` are objective dicts (``{id, statement, bloom_level?,
    terminal_id?}``) — typically a single week's TO and its child COs. Returns a
    self-contained HTML fragment (no page chrome). Deterministic.
    """
    h = max(2, min(6, int(heading_level)))

    def esc(v: Any) -> str:
        from html import escape

        return escape("" if v is None else str(v), quote=True)

    # Index COs under their parent TO (fall back to a flat list for orphans).
    cos_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    orphan_cos: List[Dict[str, Any]] = []
    for co in cos:
        pid = co.get("terminal_id") or co.get("parent_terminal")
        if isinstance(pid, str) and _TO_RE.match(pid.strip()):
            cos_by_parent.setdefault(pid.strip().upper(), []).append(co)
        else:
            orphan_cos.append(co)

    parts: List[str] = ['<div class="objective-list" data-cf-content-type="objectives">']

    def render_co_items(items: List[Dict[str, Any]]) -> None:
        parts.append('  <ul class="co-list">')
        for co in items:
            cid = esc(co.get("id", ""))
            stmt = esc(co.get("statement", ""))
            parts.append(
                f'    <li data-cf-objective-id="{cid}">'
                f"<strong>{cid}</strong> &mdash; {stmt}</li>"
            )
        parts.append("  </ul>")

    for to in tos:
        tid = esc(to.get("id", ""))
        stmt = esc(to.get("statement", ""))
        parts.append(
            f'  <div class="terminal-objective" data-cf-objective-id="{tid}">'
        )
        parts.append(f"    <h{h}>{tid} &mdash; {stmt}</h{h}>")
        children = cos_by_parent.get(
            str(to.get("id", "")).strip().upper(), []
        )
        if children:
            parts.append("    <p>Chapter objectives:</p>")
            render_co_items(children)
        parts.append("  </div>")

    if orphan_cos:
        parts.append('  <div class="terminal-objective">')
        render_co_items(orphan_cos)
        parts.append("  </div>")

    parts.append("</div>")
    return "\n".join(parts) + "\n"

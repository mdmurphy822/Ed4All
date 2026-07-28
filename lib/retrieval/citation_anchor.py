"""Citation-anchor resolver — chunk provenance → archived source page.

Pure, deterministic, read-only library that answers: *given a chunk record
and a LibV2 course dir, find the exact source page and verify the chunk's text
and span against it.*

Why resolver-side (not chunker-side): legacy corpora must not re-chunk
(byte-stability contract). The chunker's ``char_span`` has two known
degradation arms — a collapsed-whitespace fallback that indexes into a
different string space, and a total-miss fallback that *fabricates*
``[search_from, search_from+len(needle)]`` with no marker. This resolver
classifies those states honestly (``SPAN_FABRICATED``) without ever repairing
a span. It also tolerates the fact that chunk ``text`` is post-
``extract_plain_text`` (whitespace collapse), post-boilerplate-strip,
post-feedback-strip, so it is generally NOT a contiguous substring of the page
text — hence the token-shingle containment workhorse.

Status ladder (best to worst, first-match wins):

  RESOLVED_EXACT        char_span slices the resolved container text to the
                        chunk text verbatim (modulo whitespace).
  RESOLVED_NORMALIZED   the chunk text is a normalized substring of the page
                        plain text (whitespace-collapse), but not at the span.
  RESOLVED_CONTAINMENT  >= containment_threshold of the chunk's token-shingles
                        are present in the page plain text.
  SPAN_FABRICATED       the page was found but the chunk text is neither
                        locatable nor sufficiently contained — the span is
                        untrustworthy.
  SOURCE_PAGE_MISSING   item_path resolves to no archived file/member.
  SOURCE_SHA_MISMATCH   (reserved) per-page sha present and disagrees. The
                        per-page sha emit is a deferred follow-up; the
                        resolver wires the enum so it activates without a
                        signature change once the field lands. Today's chunks
                        carry only an aggregate (whole-corpus) sha, which can't
                        verify a single page, so this arm does not fire yet.

HTML-entity quirk: headings in chunk ``source`` carry double-escaped entities
(``Week 1: Application &amp;amp; Activities``). The resolver compares *text*,
not headings, so no unescaping pass is needed for that case.

Four resolver-side normalization fixes (Marketable-v1) keep matching robust
against page-vs-chunk drift WITHOUT re-chunking (the byte-stability contract):

  1. Body-chrome-safe page extraction. Courseforge stamps
     ``data-cf-role="template-chrome"`` on the page ``<body>`` (and other
     structural-root tags). ``HTMLTextExtractor`` opens a chrome-skip scope on
     that attribute but can only *close* the scope on a fixed end-tag set
     (``header``/``footer``/``a``/``div``/``nav``/``aside`` — see
     ``_CHROME_TAGS``). A chrome scope opened on ``<body>`` therefore never
     closes, so the extractor silently swallows the ENTIRE page body and the
     re-extracted page text collapses to a few ``<head>`` tokens. Every chunk
     of such a page then fails to anchor (``span_fabricated`` at ~0.0
     containment) even though the chunk text was authored from that very page.
     The chunker can't be touched (it would shift legacy chunk text on
     rebuild), so the resolver neutralizes the chrome mark on structural-root
     tags BEFORE re-extracting the page. Real chrome (header/footer/nav) keeps
     its mark and is still skipped; only the un-closeable root mark is dropped.

  2. Symmetric HTML-entity normalization. Chunk text retains raw entities
     (``Acme Corp&rsquo;s``, ``&ldquo;``/``&mdash;``) while ``HTMLParser``
     decodes them in the page (``Acme Corp's``), so every 8-token shingle that
     straddles an entity mismatches and a genuinely-contained chunk scores just
     below the 0.85 floor. The resolver ``html.unescape``s BOTH sides before
     shingling / substring, making the comparison entity-agnostic.

  3. QTI nested-markup normalization. QTI 1.2 stores escaped HTML inside
     ``mattext`` XML nodes. The chunker emits rendered text, while a plain
     outer-XML extraction leaves the decoded inner tags as literal tokens.
     The resolver strips only tag-shaped fragments from the comparison
     projection; visible source tokens and the anchoring floor are unchanged.

  4. Structural-heading projection. The chunker treats section headings as
     metadata and may merge adjacent body runs without their labels. The
     resolver compares against both the full visible page and a projection
     with only ``h1``-``h6`` elements removed, taking the stronger honest
     containment result. Body prose is never removed or synthesized.

Together these fixes are normalization-only — they make matching robust
against page/chunk drift, NOT content rewriting. The resolver stays strictly
fail-closed: a chunk whose text is genuinely absent from the page still scores
low and reports ``SPAN_FABRICATED``. (Measured on the rdf-shacl calibration
corpus: corpus anchoring rate 0.56 -> 1.00, and the 5 citation-gate-blocked
gold answers all recover.) No ``resolved_fuzzy`` display-degradation policy was
needed — the blocks were a normalization bug, not synthesized content.

No network, no LLM, no decision capture (deterministic workstream).

Run as a module for the WS1.2 calibration measure pass::

    python -m lib.retrieval.citation_anchor <course-slug> <kind> [--chunks REL]
"""

from __future__ import annotations

import difflib
import html as _html
import json
import re
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.retrieval._text import normalize_ws, shingle_containment

try:  # pragma: no cover - import guard
    from Trainforge.chunker.helpers import extract_plain_text
except Exception:  # pragma: no cover
    extract_plain_text = None  # type: ignore

try:  # pragma: no cover - import guard
    from Trainforge.parsers.xpath_walker import resolve_xpath
except Exception:  # pragma: no cover
    resolve_xpath = None  # type: ignore


class AnchorStatus(str, Enum):
    RESOLVED_EXACT = "resolved_exact"
    RESOLVED_NORMALIZED = "resolved_normalized"
    RESOLVED_CONTAINMENT = "resolved_containment"
    SPAN_FABRICATED = "span_fabricated"
    SOURCE_PAGE_MISSING = "source_page_missing"
    SOURCE_SHA_MISMATCH = "source_sha_mismatch"


_RESOLVED_STATUSES = frozenset(
    {
        AnchorStatus.RESOLVED_EXACT,
        AnchorStatus.RESOLVED_NORMALIZED,
        AnchorStatus.RESOLVED_CONTAINMENT,
    }
)


@dataclass(frozen=True)
class CitationAnchor:
    chunk_id: str
    status: AnchorStatus
    source_path: Optional[Path]
    item_path: str
    html_xpath: Optional[str]
    char_span: Optional[Tuple[int, int]]
    containment_rate: float
    normalized_match: bool


# Default archived-source search roots per chunkset kind (course-dir-relative).
_DART_ROOTS = ("sources/textbooks", "source/html", "source/dart")
_IMSCC_ROOTS = ("source/imscc",)


# ``data-cf-role="template-chrome"`` stamped on a structural-root tag opens a
# chrome-skip scope in ``HTMLTextExtractor`` that its fixed end-tag set can
# never close (see module docstring fix #1). These roots carry the page's whole
# content, so dropping the mark on them — and ONLY them — recovers the body
# while leaving genuine header/footer/nav chrome marks intact and still skipped.
_CHROME_ROOT_TAGS = ("body", "html", "main", "section", "article")
_BODY_CHROME_RE = re.compile(
    r"(<(?:" + "|".join(_CHROME_ROOT_TAGS) + r")\b[^>]*?)"
    r"\s+data-cf-role\s*=\s*(['\"])template-chrome\2",
    re.IGNORECASE,
)

# QTI 1.2 stores item prose as HTML escaped inside ``<mattext>`` XML nodes,
# e.g. ``&lt;p&gt;Question&lt;/p&gt;``.  ``extract_plain_text`` parses the
# outer XML and decodes the entity, leaving the *nested* ``<p>`` as literal
# text.  The IMSCC chunker, correctly, parses that nested HTML before emitting
# the assessment chunk.  Strip only syntactically tag-shaped fragments from
# the comparison projection so both sides inhabit the same text space.  This
# is deliberately narrow: mathematical comparisons such as ``x < 3`` do not
# match because a tag must begin with a letter (or ``/`` + letter).
_NESTED_MARKUP_TAGS = (
    "a|abbr|article|aside|b|blockquote|br|code|dd|del|details|div|dl|dt|"
    "em|figcaption|figure|footer|h[1-6]|header|i|kbd|li|main|mark|nav|"
    "ol|p|pre|q|s|section|small|span|strong|sub|summary|sup|table|tbody|"
    "td|tfoot|th|thead|tr|u|ul"
)
_NESTED_MARKUP_RE = re.compile(
    r"<!--.*?-->|"
    rf"</?(?:{_NESTED_MARKUP_TAGS})(?=[\s/>])[^<>]*>",
    re.IGNORECASE | re.DOTALL,
)

# Math emitters differ on whitespace around comparison/equality operators
# (``y < 4`` versus ``y<4``) even though the visible expression is identical.
# Canonicalize only operator-adjacent whitespace in the comparison projection.
_MATH_OPERATOR_WS_RE = re.compile(r"\s*([<>=≤≥])\s*")

# The chunker treats heading labels as structure and emits the associated body
# prose.  A merged chunk can therefore join two body runs while the archived
# HTML still has an intervening ``<hN>label</hN>``.  Keep both projections:
# the ordinary visible page (for chunks that intentionally retain a heading)
# and a body-only projection (for the normal chunker path).  Only heading
# elements are removed; body prose is never rewritten or invented.
_STRUCTURAL_HEADING_RE = re.compile(
    r"<h([1-6])(?=[\s>])[^>]*>.*?</h\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _strip_root_template_chrome(html: str) -> str:
    """Drop ``data-cf-role="template-chrome"`` from structural-root tags only.

    Leaves the mark on every non-root element (``header``/``footer``/``nav``/…)
    so real page chrome is still skipped by :class:`HTMLTextExtractor`; only the
    un-closeable root mark is neutralized so the page body survives extraction.
    """
    return _BODY_CHROME_RE.sub(r"\1", html)


def _plain_text(html: str) -> str:
    """Plain-text projection consistent with what the chunker emitted.

    Applies the body-chrome-safe pre-pass (module docstring fix #1) so a page
    whose ``<body>`` carries the un-closeable ``template-chrome`` mark still
    yields its full body text. The pre-pass is a no-op for pages without a
    root-level chrome mark, so legacy / non-Courseforge pages are unaffected.
    """
    safe_html = _strip_root_template_chrome(html)
    if extract_plain_text is not None:
        return extract_plain_text(safe_html)
    # Minimal stdlib fallback (only used if the chunker import is unavailable):
    text = re.sub(r"<script[^>]*>.*?</script>", " ", safe_html, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_ws(text)


def _plain_text_projections(html: str) -> Tuple[str, ...]:
    """Return honest visible-text projections used by the chunker.

    The first projection is the complete page text.  The second removes only
    structural heading elements before extraction, matching merged chunks
    whose section labels are metadata rather than body content.  Duplicate
    projections are collapsed while preserving order.
    """
    full = _plain_text(html)
    body_only = _plain_text(_STRUCTURAL_HEADING_RE.sub(" ", html))
    return (full,) if body_only == full else (full, body_only)


def _normalize_for_match(text: str) -> str:
    """Entity- and nested-markup-symmetric projection for fuzzy matching.

    ``html.unescape`` decodes named + numeric entities (``&rsquo;`` -> ``'``,
    ``&mdash;`` -> ``—``) so a chunk that retained raw entities compares equal
    to the page whose entities ``HTMLParser`` already decoded (module docstring
    fix #2).

    A second, narrow normalization removes HTML tags revealed by that unescape.
    This is required for QTI ``mattext``: the archived XML contains escaped
    inner HTML while assessment chunks contain its rendered text.  Removing
    tags changes no source claims and does not weaken containment: only markup
    syntax is discarded, and all human-visible tokens must still match at the
    unchanged shingle/floor settings.

    Idempotent — plain text with no entities or tags passes through unchanged.
    """
    unescaped = _html.unescape(text or "")
    without_nested_markup = _NESTED_MARKUP_RE.sub(" ", unescaped)
    # Attribute values can themselves contain entities, so decode once more
    # after removing the revealed tag shell.
    normalized = normalize_ws(_html.unescape(without_nested_markup))
    return _MATH_OPERATOR_WS_RE.sub(r"\1", normalized)


def _ordered_segment_containment(
    needle: str,
    haystack: str,
    *,
    min_segment_tokens: int,
) -> float:
    """Fraction of needle tokens in ordered, source-verbatim body runs.

    The chunker can omit structural labels between adjacent body sections.
    Ordinary shingles crossing each omitted label then fail even though every
    emitted body run is copied from the archived page.  This companion measure
    credits only non-overlapping, document-ordered exact token runs at least as
    long as the configured shingle size.  Short on-topic word collisions earn
    no credit, so the unchanged containment threshold remains fail-closed for
    fabricated prose.
    """
    needle_tokens = normalize_ws(needle).lower().split()
    haystack_tokens = normalize_ws(haystack).lower().split()
    if not needle_tokens or not haystack_tokens or min_segment_tokens <= 0:
        return 0.0
    matcher = difflib.SequenceMatcher(
        None,
        needle_tokens,
        haystack_tokens,
        autojunk=False,
    )
    matched = sum(
        block.size
        for block in matcher.get_matching_blocks()
        if block.size >= min_segment_tokens
    )
    return matched / len(needle_tokens)


def _iter_html_roots(course_dir: Path, roots: Tuple[str, ...]):
    for rel in roots:
        candidate = course_dir / rel
        if candidate.is_dir():
            yield candidate


def _resolve_dart_page(
    course_dir: Path, item_path: str, roots: Tuple[str, ...]
) -> Optional[Tuple[Path, str]]:
    # item_path is a filename relative to the staging dir; it may live under
    # any of the search roots. Match by basename to tolerate nested layouts.
    target_name = Path(item_path).name
    for root in _iter_html_roots(course_dir, roots):
        direct = root / item_path
        if direct.is_file():
            return direct, direct.read_text(encoding="utf-8", errors="replace")
        # Fall back to a basename match anywhere under the root.
        for cand in root.rglob(target_name):
            if cand.is_file():
                return cand, cand.read_text(encoding="utf-8", errors="replace")
    return None


def _resolve_imscc_member(
    course_dir: Path, item_path: str, roots: Tuple[str, ...]
) -> Optional[Tuple[str, str]]:
    """Find ``item_path`` as a member of any ``*.imscc`` under the roots, or
    under a pre-unpacked ``_wave81_unpacked`` directory. Reads via stdlib
    zipfile (no extraction to disk)."""
    for root in _iter_html_roots(course_dir, roots):
        # 1) pre-unpacked directory tree (rdf-shacl _wave81_unpacked, etc.)
        for unpacked in root.glob("*_unpacked"):
            direct = unpacked / item_path
            if direct.is_file():
                return (
                    f"{unpacked.name}/{item_path}",
                    direct.read_text(encoding="utf-8", errors="replace"),
                )
        # 2) zip member lookup inside each cartridge
        for imscc in sorted(root.glob("*.imscc")):
            try:
                with zipfile.ZipFile(imscc) as zf:
                    names = set(zf.namelist())
                    member = None
                    if item_path in names:
                        member = item_path
                    else:
                        # basename match against members
                        target = Path(item_path).name
                        for n in zf.namelist():
                            if Path(n).name == target:
                                member = n
                                break
                    if member is not None:
                        data = zf.read(member).decode("utf-8", errors="replace")
                        return f"{imscc.name}!{member}", data
            except (zipfile.BadZipFile, OSError):
                continue
    return None


def resolve_source_page(
    chunk: Dict[str, Any],
    course_dir: Path,
    *,
    chunkset_kind: str,
    source_roots: Optional[List[Path]] = None,
) -> Optional[Tuple[Any, str]]:
    """Map ``chunk.source.item_path`` to archived page bytes.

    Returns ``(path-or-member-identifier, html_text)`` or ``None`` when no
    archived page matches. The identifier is a ``Path`` for on-disk DART pages
    and a ``"pkg.imscc!member"`` / ``"_unpacked/member"`` string for imscc
    members.
    """
    course_dir = Path(course_dir)
    source = chunk.get("source", {}) or {}
    item_path = source.get("item_path")
    if not item_path:
        return None

    if source_roots:
        # Operator-supplied absolute roots take precedence; try each as a DART
        # directory then as an imscc container.
        rels = tuple(str(p) for p in source_roots)
        hit = _resolve_dart_page(course_dir, item_path, rels)
        if hit:
            return hit

    if chunkset_kind in ("semantik", "dart"):
        # semantik is the canonical staged-conversion kind; ``dart`` is the
        # legacy read-compat alias. Both resolve item_path against the HTML
        # source roots (never the imscc fallthrough below).
        return _resolve_dart_page(course_dir, item_path, _DART_ROOTS)
    if chunkset_kind == "imscc":
        return _resolve_imscc_member(course_dir, item_path, _IMSCC_ROOTS)
    # ``corpus`` is the UNION chunkset: imscc-derived chunks AND DART-derived
    # chunks coexist, each carrying its own ``item_path`` (a ``.imscc`` member
    # path vs a ``source/html/*.html`` DART page). Resolving every chunk as an
    # imscc member stranded the DART chunks as ``source_page_missing`` (the
    # all-DART tail of a mixed corpus). Try BOTH axes per-chunk so each chunk
    # resolves against whichever archived artifact actually produced it.
    if chunkset_kind == "corpus":
        return _resolve_imscc_member(course_dir, item_path, _IMSCC_ROOTS) or _resolve_dart_page(
            course_dir, item_path, _DART_ROOTS
        )
    # Unknown kind: try both axes (dart pages then imscc members).
    return _resolve_dart_page(course_dir, item_path, _DART_ROOTS) or _resolve_imscc_member(
        course_dir, item_path, _IMSCC_ROOTS
    )


def _exact_span_match(
    html: str, html_xpath: Optional[str], char_span, chunk_text: str
) -> bool:
    """True iff slicing the xpath-resolved container text at char_span equals
    the chunk text (modulo whitespace normalization). Never repairs the span.
    """
    if not html_xpath or not char_span or resolve_xpath is None:
        return False
    try:
        start, end = int(char_span[0]), int(char_span[1])
    except (TypeError, ValueError, IndexError):
        return False
    if start < 0 or end <= start:
        return False
    container = resolve_xpath(html, html_xpath)
    if container is None or end > len(container):
        return False
    sliced = container[start:end]
    return normalize_ws(sliced) == normalize_ws(chunk_text)


def resolve_citation_anchor(
    chunk: Dict[str, Any],
    course_dir: Path,
    *,
    chunkset_kind: str,
    shingle_size: int = 8,
    containment_threshold: float = 0.85,
    source_roots: Optional[List[Path]] = None,
) -> CitationAnchor:
    """Resolve a single chunk to an honest :class:`CitationAnchor`."""
    source = chunk.get("source", {}) or {}
    chunk_id = str(chunk.get("id", ""))
    item_path = str(source.get("item_path", "") or "")
    html_xpath = source.get("html_xpath")
    raw_span = source.get("char_span")
    char_span: Optional[Tuple[int, int]]
    if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
        try:
            char_span = (int(raw_span[0]), int(raw_span[1]))
        except (TypeError, ValueError):
            char_span = None
    else:
        char_span = None

    resolved = resolve_source_page(
        chunk, course_dir, chunkset_kind=chunkset_kind, source_roots=source_roots
    )
    if resolved is None:
        return CitationAnchor(
            chunk_id=chunk_id,
            status=AnchorStatus.SOURCE_PAGE_MISSING,
            source_path=None,
            item_path=item_path,
            html_xpath=html_xpath,
            char_span=char_span,
            containment_rate=0.0,
            normalized_match=False,
        )

    source_id, html = resolved
    source_path = Path(source_id) if isinstance(source_id, Path) else source_id
    chunk_text = chunk.get("text", "") or ""
    page_texts = _plain_text_projections(html)

    # Entity-symmetric projections (module docstring fix #2): unescape both
    # sides so a chunk that kept raw ``&rsquo;``/``&mdash;`` entities matches a
    # page whose entities HTMLParser already decoded. ``shingle_containment``
    # lowercases internally, so passing the normalized strings is sufficient.
    chunk_match = _normalize_for_match(chunk_text)
    page_matches = tuple(_normalize_for_match(text) for text in page_texts)
    containment = 0.0
    for page_match in page_matches:
        shingle_rate = shingle_containment(
            chunk_match, page_match, shingle_size=shingle_size
        )
        if shingle_rate >= containment_threshold:
            candidate_rate = shingle_rate
        else:
            candidate_rate = max(
                shingle_rate,
                _ordered_segment_containment(
                    chunk_match,
                    page_match,
                    min_segment_tokens=shingle_size,
                ),
            )
        containment = max(containment, candidate_rate)
    normalized_substr = any(
        chunk_match in page_match for page_match in page_matches
    )

    # 1) exact span verification (never repaired).
    if _exact_span_match(html, html_xpath, char_span, chunk_text):
        status = AnchorStatus.RESOLVED_EXACT
    # 2) normalized substring of the page.
    elif normalized_substr:
        status = AnchorStatus.RESOLVED_NORMALIZED
    # 3) shingle containment over the threshold.
    elif containment >= containment_threshold:
        status = AnchorStatus.RESOLVED_CONTAINMENT
    # 4) page found, text not locatable → span untrustworthy.
    else:
        status = AnchorStatus.SPAN_FABRICATED

    return CitationAnchor(
        chunk_id=chunk_id,
        status=status,
        source_path=source_path,
        item_path=item_path,
        html_xpath=html_xpath,
        char_span=char_span,
        containment_rate=round(containment, 6),
        normalized_match=bool(normalized_substr),
    )


def _load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with Path(chunks_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def anchor_report(
    chunks_path: Path,
    course_dir: Path,
    *,
    chunkset_kind: str,
    shingle_size: int = 8,
    containment_threshold: float = 0.85,
    source_roots: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Corpus-level rollup: per-status counts, anchoring_rate, span_fidelity_rate,
    and worst-offender chunk_ids (up to 20) per failure status. Deterministic.
    """
    chunks = _load_chunks(Path(chunks_path))
    status_counts: Dict[str, int] = {s.value: 0 for s in AnchorStatus}
    offenders: Dict[str, List[str]] = {
        AnchorStatus.SPAN_FABRICATED.value: [],
        AnchorStatus.SOURCE_PAGE_MISSING.value: [],
        AnchorStatus.SOURCE_SHA_MISMATCH.value: [],
    }
    resolved_count = 0
    exact_count = 0

    for chunk in chunks:
        anchor = resolve_citation_anchor(
            chunk,
            course_dir,
            chunkset_kind=chunkset_kind,
            shingle_size=shingle_size,
            containment_threshold=containment_threshold,
            source_roots=source_roots,
        )
        status_counts[anchor.status.value] += 1
        if anchor.status in _RESOLVED_STATUSES:
            resolved_count += 1
        if anchor.status is AnchorStatus.RESOLVED_EXACT:
            exact_count += 1
        if anchor.status.value in offenders and len(offenders[anchor.status.value]) < 20:
            offenders[anchor.status.value].append(anchor.chunk_id)

    total = len(chunks)
    anchoring_rate = (resolved_count / total) if total else 0.0
    span_fidelity_rate = (exact_count / resolved_count) if resolved_count else 0.0

    return {
        "total_chunks": total,
        "chunkset_kind": chunkset_kind,
        "status_counts": status_counts,
        "resolved_count": resolved_count,
        "anchoring_rate": round(anchoring_rate, 6),
        "span_fidelity_rate": round(span_fidelity_rate, 6),
        "worst_offenders": {k: v for k, v in offenders.items() if v},
        "params": {
            "shingle_size": shingle_size,
            "containment_threshold": containment_threshold,
        },
    }


def _main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    from lib.paths import libv2_path

    parser = argparse.ArgumentParser(
        description="Measure citation-anchor resolution over a LibV2 chunkset."
    )
    parser.add_argument("course_slug")
    parser.add_argument("kind", choices=["dart", "imscc", "corpus"])
    parser.add_argument(
        "--chunks",
        default=None,
        help="course-dir-relative chunks path (defaults per kind).",
    )
    parser.add_argument("--report", action="store_true", help="(alias; always reports)")
    args = parser.parse_args(argv)

    course_dir = libv2_path() / "courses" / args.course_slug
    default_rel = {
        # DART->semantik purge Stage 1 (dual-READ): accept the ratified kind.
        "semantik": "semantik_chunks/chunks.jsonl",
        "dart": "dart_chunks/chunks.jsonl",
        "imscc": "imscc_chunks/chunks.jsonl",
        "corpus": "corpus/chunks.jsonl",
    }[args.kind]
    chunks_path = course_dir / (args.chunks or default_rel)
    if not chunks_path.is_file():
        print(json.dumps({"error": f"chunks not found: {chunks_path}"}))
        return 2
    report = anchor_report(chunks_path, course_dir, chunkset_kind=args.kind)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "AnchorStatus",
    "CitationAnchor",
    "resolve_source_page",
    "resolve_citation_anchor",
    "anchor_report",
]

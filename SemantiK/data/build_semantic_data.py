"""Build BERT-Semantic (Phase 3c) training data.

Span-level multi-head classifier on PDF blocks. Cascaded from
BERT-Structure: each row carries Structure's predicted top-k labels as
input features, in addition to the text + layout side-channel.

Heads (both span-level):
    * doc_role           7-class    {title, author, body, citation,
                                     footer, legal, metadata}
                                     — extracted from raw_source_html
                                     (Phase 3c augmentation; the
                                     emit_html.emit() output strips the
                                     semantic markup we need).
                                     NOTE: The original 8-class vocab
                                     included ``abstract``; it was
                                     dropped at 0.16% prevalence (183
                                     rows on a 116k corpus) — too sparse
                                     to train a head on. Abstract
                                     content is now relabeled as
                                     ``body`` since structural layout
                                     (paragraph) is what drives the
                                     WCAG emit, not whether a paragraph
                                     happens to be the abstract.
    * boilerplate_flag   binary     1 if span is repeated/trivial chrome
                                     (page numbers, copyright lines,
                                     arxiv banners, "Continued from…").
                                     Heuristic-derived; can co-occur
                                     with any doc_role.

Cascade input (Structure → Semantic):
    Pre-run the trained Structure adapter on every block to attach an
    8-dim numeric vector per row:
        [P(structural_role)x6, P(is_heading=1), P(table_region=1)]
    Concatenated with the 20-dim layout vector (28-dim total) and fed
    through the layout MLP. This is the "teacher-forced" inputs path —
    scheduled-sampling at training time will swap to predicted top-k
    in the final 10% of epochs.

Source pipeline (per pair):
    1. Walk pair files: ``data/pairs/<source>/*.json``.
    2. For each pair, run ``extract_shared`` on local_pdf (cached) OR
       render output_html via Playwright.
    3. featurize_from_shared → list[FeatureBlock].
    4. Run trained Structure adapter to get per-block softmax
       probabilities for all 4 heads — packed into the cascade vector.
    5. Walk raw_source_html with a SOURCE-SPECIFIC extractor (arxiv
       ar5iv, wikipedia Parsoid, gutenberg, openstax,
       federal_register XML) to emit labeled blocks.
    6. Align each pypdfium2 merged block to a labeled HTML block via
       Jaccard.
    7. Emit row: (text, layout_vec, structure_cascade_vec, doc_role,
       boilerplate_flag).

Outputs:
    data/semantic_dataset/{train,val,test}.jsonl
    data/semantic_dataset/coverage_report.json

Network: NONE. Uses local pair files + cached Structure adapter +
cached extractor output.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from dart_semantic.extract_shared import extract_shared, extract_shared_cached
from dart_semantic.text_utils import jaccard_overlap
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool

# Reuse Phase 3b layout helpers — the cascade input rides the SAME 20-dim
# layout vector (order/dim must match exactly so the trained Structure
# adapter can be invoked over the same features the row carries).
from data.balance import add_cap_args, apply_caps_and_report
from data.build_structure_data import (
    LAYOUT_FEATURE_DIM,
    LAYOUT_FEATURE_NAMES,
    _block_in_any_table,
    compute_span_layout_features,
)


# ---------------------------------------------------------------------------
# Label vocabulary
# ---------------------------------------------------------------------------

# 7 doc-role classes — see header docstring. Keep order stable: head
# logits map to DOC_ROLE_NAMES[i] via DOC_ROLE_LIST[i].
# Note: ``abstract`` was removed from the original 8-class vocab at
# 0.16% prevalence; abstract content is now emitted as ``body``. This
# shifts integer IDs (body 3->2, citation 4->3, footer 5->4, legal 6->5,
# metadata 7->6) — any pre-shift JSONL must be regenerated.
DOC_ROLE_NAMES = (
    "title",
    "author",
    "body",
    "citation",
    "footer",
    "legal",
    "metadata",
)
DOC_ROLE_TO_ID = {r: i for i, r in enumerate(DOC_ROLE_NAMES)}
NUM_DOC_ROLES = len(DOC_ROLE_NAMES)


# ---------------------------------------------------------------------------
# Boilerplate detector — heuristic. Independent of doc_role.
# ---------------------------------------------------------------------------

import re

_PAGE_NUM_RE = re.compile(r"^\s*\d+\s*(/\s*\d+)?\s*$")
_ARXIV_BANNER_RE = re.compile(r"arxiv:\s*\d+\.\d+", re.I)
_DOI_RE = re.compile(r"doi:?\s*10\.\d+/", re.I)
_COPYRIGHT_RE = re.compile(r"copyright|©|\(c\)\s*\d{4}|all rights reserved", re.I)
_CONTINUED_RE = re.compile(r"continued (on|from)|see (next|previous) page", re.I)
_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_LICENSE_RE = re.compile(
    r"creative commons|cc by(-[a-z]+)*|gpl-?\d|"
    r"mit license|apache 2\.0|public domain",
    re.I,
)


def is_boilerplate(text: str, doc_role: str) -> int:
    """Heuristic 1/0 — does this span look like repeated chrome?"""
    t = text.strip()
    if not t:
        return 0
    if len(t) <= 80 and _PAGE_NUM_RE.match(t):
        return 1
    if _ARXIV_BANNER_RE.search(t):
        return 1
    if _DOI_RE.search(t) and len(t) < 200:
        return 1
    if _COPYRIGHT_RE.search(t):
        return 1
    if _LICENSE_RE.search(t):
        return 1
    if _CONTINUED_RE.search(t):
        return 1
    # URL-only or email-only short spans
    if len(t) <= 100 and (_URL_RE.fullmatch(t) or _EMAIL_RE.fullmatch(t)):
        return 1
    # All artifacts / footers / legal / metadata are boilerplate by
    # default unless the text suggests substantive content.
    if doc_role in ("footer", "legal", "metadata"):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Source-specific raw-HTML walkers
# ---------------------------------------------------------------------------
# Each returns list[{text, doc_role, source}] in document order. The
# caller aligns these to PDF-extracted blocks via Jaccard.


def _clean_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def extract_arxiv_blocks(raw_html: str) -> list[dict]:
    """ar5iv source HTML — semantic markup is class-based.

    Mapping:
      .ltx_title_document   -> title
      .ltx_authors,         -> author (whole authors block)
        .ltx_personname     -> author (per-name; we use the wrapper)
      .ltx_abstract         -> body  (relabeled — abstract class dropped
                                       from vocab at 0.16% prevalence)
      .ltx_para             -> body (each para is one block; paras
                                     inside ltx_abstract are skipped to
                                     avoid double-counting with the
                                     wrapper-level emit above)
      .ltx_bibitem          -> citation (one per reference entry)
      .ltx_page_footer      -> footer

    metadata: arxiv banner / DOI lines fall under boilerplate detection
    instead of a separate HTML class.
    """
    soup = BeautifulSoup(raw_html, "lxml")
    out: list[dict] = []

    # Title — usually first
    for el in soup.find_all(class_="ltx_title_document"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "title"})

    # Author — emit ONE row per ltx_authors wrapper (the full author
    # paragraph); per-personname rows would over-count when an author
    # block has many co-authors.
    for el in soup.find_all(class_="ltx_authors"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "author"})

    # Abstract content — now relabeled as body. Emit at the wrapper
    # level so we don't double-count with .ltx_para descendants below
    # (which we exclude from the body sweep when nested in ltx_abstract).
    for el in soup.find_all(class_="ltx_abstract"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "body"})

    # Body — every ltx_para is a paragraph block. Skip ones nested in
    # ltx_abstract (already captured at wrapper level above as body),
    # ltx_bibliography / ltx_bibitem (citations), ltx_authors (authors),
    # ltx_page_footer (footer).
    skipped_in = (
        "ltx_abstract",
        "ltx_bibliography",
        "ltx_bibitem",
        "ltx_authors",
        "ltx_page_footer",
    )
    for el in soup.find_all(class_="ltx_para"):
        if any(
            p.get("class") and any(c in p.get("class", []) for c in skipped_in) for p in el.parents
        ):
            continue
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "body"})

    # Citations — one row per bibitem
    for el in soup.find_all(class_="ltx_bibitem"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "citation"})

    # Footer
    for el in soup.find_all(class_="ltx_page_footer"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "footer"})

    return out


_WIKI_REFERENCE_CLASS_RE = re.compile(r"reference|references|cite", re.I)
_WIKI_NAV_CLASS_RE = re.compile(
    r"navbox|navigation|hatnote|infobox|"
    r"sidebar|metadata",
    re.I,
)


def extract_wikipedia_blocks(raw_html: str) -> list[dict]:
    """Wikipedia Parsoid — mostly body + citations.

    Section-aware where possible:
      <section data-mw-section-id="0">  -> body (lead section; was
                                           previously labeled "abstract"
                                           but the lead is structurally
                                           body prose and the abstract
                                           class was dropped from vocab)
      <h1>                              -> title
      .hatnote, .infobox, .shortdescription -> metadata
      .navbox                           -> footer (navigation chrome)
      .reference, ol.references, <cite> -> citation
      <p> elsewhere                     -> body
    """
    soup = BeautifulSoup(raw_html, "lxml")
    out: list[dict] = []

    # Title — Parsoid returns body content without an <h1>; the page
    # title lives in <head><title>.
    if soup.title and soup.title.string:
        text = _clean_text(soup.title.string)
        if text:
            out.append({"text": text, "doc_role": "title"})

    # Lead section (data-mw-section-id="0") — emitted as body. Previously
    # this routed to "abstract"; that class is gone (see DOC_ROLE_NAMES
    # comment). We still record the IDs so the generic <p> sweep below
    # doesn't double-emit them.
    lead = soup.find("section", attrs={"data-mw-section-id": "0"})
    lead_paragraph_ids = set()
    if lead is not None:
        for p in lead.find_all("p"):
            text = _clean_text(p.get_text(" "))
            if text:
                out.append({"text": text, "doc_role": "body"})
                lead_paragraph_ids.add(id(p))

    # hatnote / shortdescription / infobox -> metadata
    for el in soup.find_all(class_=re.compile(r"hatnote|shortdescription|infobox", re.I)):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "metadata"})

    # navbox -> footer (article-bottom navigation chrome)
    for el in soup.find_all(class_=re.compile(r"navbox", re.I)):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "footer"})

    # Body — top-level <p> tags not in references/infobox/navigation
    # AND not already emitted via the lead section (would double-emit).
    for el in soup.find_all("p"):
        if id(el) in lead_paragraph_ids:
            continue
        skip = False
        for p in el.parents:
            if not isinstance(p, Tag):
                continue
            cls = " ".join(p.get("class", []))
            role = p.get("role", "")
            if (
                _WIKI_REFERENCE_CLASS_RE.search(cls)
                or _WIKI_NAV_CLASS_RE.search(cls)
                or role == "navigation"
            ):
                skip = True
                break
        if skip:
            continue
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "body"})

    # Citations — <ol class="references"> contents
    for ol in soup.find_all("ol", class_=_WIKI_REFERENCE_CLASS_RE):
        for li in ol.find_all("li"):
            text = _clean_text(li.get_text(" "))
            if text:
                out.append({"text": text, "doc_role": "citation"})
    # Also <cite> tags
    for el in soup.find_all("cite"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "citation"})

    return out


def extract_gutenberg_blocks(raw_html: str) -> list[dict]:
    """Project Gutenberg — body + license preamble.

    Mapping:
      <h1>                                       -> title (book title)
      <p> outside the boilerplate header/footer   -> body
      "*** START OF THIS PROJECT GUTENBERG…"     -> legal (boilerplate)
      "*** END OF THIS PROJECT GUTENBERG…"       -> legal (boilerplate)
    """
    soup = BeautifulSoup(raw_html, "lxml")
    out: list[dict] = []

    h1 = soup.find("h1")
    if h1:
        text = _clean_text(h1.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "title"})

    # Body: <p> tags. Gutenberg legal preamble is usually a long single
    # block before "*** START …" markers; mark those as legal.
    started = False
    in_legal = True  # everything before *** START is legal preamble
    for el in soup.find_all(["p", "pre", "div"]):
        text = _clean_text(el.get_text(" "))
        if not text:
            continue
        upper = text.upper()
        if "*** START OF" in upper and "PROJECT GUTENBERG" in upper:
            started = True
            in_legal = False
            out.append({"text": text, "doc_role": "legal"})
            continue
        if "*** END OF" in upper and "PROJECT GUTENBERG" in upper:
            in_legal = True
            out.append({"text": text, "doc_role": "legal"})
            continue
        if in_legal:
            out.append({"text": text, "doc_role": "legal"})
        else:
            out.append({"text": text, "doc_role": "body"})

    return out


def extract_openstax_blocks(raw_html: str) -> list[dict]:
    """OpenStax — body-heavy; license footer captured as legal."""
    soup = BeautifulSoup(raw_html, "lxml")
    out: list[dict] = []

    h1 = soup.find("h1")
    if h1:
        text = _clean_text(h1.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "title"})

    for el in soup.find_all("p"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "body"})

    # Footer / license blocks
    for el in soup.find_all(["footer"]):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "legal"})

    return out


def extract_federal_register_blocks(raw_xml: str) -> list[dict]:
    """Federal Register XML — heavy boilerplate / legal / metadata."""
    soup = BeautifulSoup(raw_xml, "lxml-xml")
    out: list[dict] = []

    # AGENCY / DOCTYPE / RIN / DATES are all metadata
    for tag in ("AGENCY", "DOCTYPE", "ACT", "DATES", "FURINF", "AGY", "RIN"):
        for el in soup.find_all(tag):
            text = _clean_text(el.get_text(" "))
            if text:
                out.append({"text": text, "doc_role": "metadata"})

    # Subject / TITLE → title
    for tag in ("SUBJECT", "TITLE"):
        for el in soup.find_all(tag):
            text = _clean_text(el.get_text(" "))
            if text:
                out.append({"text": text, "doc_role": "title"})

    # Body paragraphs
    for el in soup.find_all("P"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "body"})

    # Footnotes / footers
    for el in soup.find_all(["FTNT", "FRDOC", "FILED"]):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "footer"})

    return out


def extract_cfr_blocks(raw_xml: str) -> list[dict]:
    """govinfo.gov CFR PART XML — strong source for legal/footer/metadata.

    Emits blocks in DOCUMENT ORDER (depth-first walk). The Phase 1
    aligner uses a forward-only sliding cursor over html_blocks; if
    blocks aren't in document order the cursor blows past metadata
    while looking for body and the rare classes never align.

    Mapping (per parse_cfr.py contract):

      OENOTICE / GPO / SUDOCS / BTITLE  -> legal (front-matter notice)
      AUTH / SOURCE / EFFDATE           -> metadata (citation lines)
      EAR ("Pt. 1")                     -> metadata
      SECTNO ("§ 1.1")                  -> metadata
      CFRNO / CFRTITLE / TITLENUM       -> metadata
      CITA / CITEP / EDNOTE             -> footer (edition / citation)
      FTNT / NOTE                       -> footer (footnotes)
      HD SOURCE="HED" (top-level PART)  -> title
      SUBJECT (in SECTION)              -> title
      P / FP                            -> body (regulation text)
    """
    soup = BeautifulSoup(raw_xml, "lxml-xml")
    out: list[dict] = []

    LEGAL_CONTAINERS = {"OENOTICE", "GPO", "SUDOCS"}
    METADATA_CONTAINERS = {"AUTH", "SOURCE", "EFFDATE"}
    FOOTER_CONTAINERS = {"FTNT", "NOTE"}
    METADATA_LEAF = {"EAR", "SECTNO", "CFRNO", "CFRTITLE", "TITLENUM", "CHAPNO", "AGENCY", "AGY"}
    FOOTER_LEAF = {"CITA", "CITEP", "EDNOTE"}
    SKIP_CONTAINERS = {
        "FMTR",
        "BMTR",
        "EXPL",
        "EXPLA",
        "ALPHLIST",
        "LSA",
        "FAIDS",
        "FR",
        "TOC",
        "CFRTOC",
        "CONTENTS",
        "TITLEPG",
        "BTITLE",
    }

    # The root may be CFRDOC (when fetcher merged FMTR + PART) or a
    # bare PART. Either way, descend.
    root = soup.find("CFRDOC") or soup
    seen_first_hed = False

    def emit(text: str, role: str) -> None:
        text = _clean_text(text)
        if text:
            out.append({"text": text, "doc_role": role})

    def walk(
        el,
        *,
        in_legal: bool = False,
        in_metadata: bool = False,
        in_footer: bool = False,
        in_skip: bool = False,
    ) -> None:
        nonlocal seen_first_hed
        if not hasattr(el, "name") or el.name is None:
            return
        tag = el.name.upper()

        if tag in LEGAL_CONTAINERS:
            for p in el.find_all(["P", "FP"]):
                emit(p.get_text(" "), "legal")
            return
        if tag in METADATA_CONTAINERS:
            emit(el.get_text(" "), "metadata")
            return
        if tag in FOOTER_CONTAINERS:
            emit(el.get_text(" "), "footer")
            return
        if tag in METADATA_LEAF:
            emit(el.get_text(" "), "metadata")
            return
        if tag in FOOTER_LEAF:
            emit(el.get_text(" "), "footer")
            return
        if tag in SKIP_CONTAINERS:
            # Walk children with skip flag set so we don't accidentally
            # harvest body P from inside, but we DO want to find legal
            # / metadata containers nested inside (e.g., OENOTICE inside
            # BTITLE inside FMTR). Recurse with in_skip flag.
            for child in el.children:
                walk(
                    child,
                    in_skip=True,
                    in_legal=in_legal,
                    in_metadata=in_metadata,
                    in_footer=in_footer,
                )
            return

        # SECTION emits its SUBJECT as title BEFORE descending into body
        if tag == "SECTION":
            subj = el.find("SUBJECT", recursive=False)
            if subj is not None:
                emit(subj.get_text(" "), "title")
            for child in el.children:
                walk(
                    child,
                    in_legal=in_legal,
                    in_metadata=in_metadata,
                    in_footer=in_footer,
                    in_skip=in_skip,
                )
            return

        if tag == "SUBJECT":
            # Already handled by SECTION parent; if at root level,
            # treat as title.
            return

        if tag == "HD":
            source = (el.get("SOURCE") or "").upper()
            if source == "HED" and not seen_first_hed:
                seen_first_hed = True
                # First HED of the PART is the title.
                if not in_skip:
                    emit(el.get_text(" "), "title")
                return
            # Other HDs: skip silently (parser puts subjects there).
            return

        if tag in ("P", "FP"):
            if in_skip:
                return
            emit(el.get_text(" "), "body")
            return

        # Generic descend
        for child in el.children:
            walk(
                child,
                in_legal=in_legal,
                in_metadata=in_metadata,
                in_footer=in_footer,
                in_skip=in_skip,
            )

    walk(root)
    return out


def extract_pmc_blocks(raw_xml: str) -> list[dict]:
    """PubMed Central Open Access subset — JATS XML.

    NCBI's PMC Open Access "Commercial Use" subset only ships articles
    licensed CC-BY / CC-BY-SA / CC0 (per https://www.ncbi.nlm.nih.gov/pmc/
    tools/openftlist/ — the OA Commercial subset is the licence-filtered
    bucket). The fetcher MUST drop articles whose <license-type> is not
    in {open-access, cc-by, cc-by-sa, cc0}; this extractor assumes that
    has already been done at pair-build time.

    JATS mapping (per NLM Journal Archiving DTD):

      <article-title>                          -> title
      <contrib-group> / <name> wrapper          -> author (one row per
                                                   contrib-group; per-name
                                                   would over-count when
                                                   there are 30 co-authors)
      <abstract>                               -> body (relabeled — abstract
                                                   class dropped from vocab)
      <sec> // <p>                             -> body (paragraph)
      <ref-list> // <ref>                      -> citation (one per ref)
      <fn> (footnote), <notes>                 -> footer
      <copyright-statement>, <license-p>       -> legal
      <article-id>, <pub-date>, <volume>,      -> metadata
        <issue>, <fpage>, <lpage>,
        <elocation-id>, <article-categories>,
        <journal-title>, <journal-id>,
        <publisher-name>, <issn>
    """
    soup = BeautifulSoup(raw_xml, "lxml-xml")
    out: list[dict] = []

    # ---- title -----------------------------------------------------------
    for el in soup.find_all("article-title"):
        # Skip article-titles nested inside <ref> (those are cited-work
        # titles, captured under citation).
        if any(p.name == "ref" for p in el.parents if hasattr(p, "name")):
            continue
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "title"})

    # ---- metadata --------------------------------------------------------
    META_TAGS = (
        "journal-id",
        "journal-title",
        "publisher-name",
        "issn",
        "article-id",
        "pub-date",
        "volume",
        "issue",
        "fpage",
        "lpage",
        "elocation-id",
        "article-categories",
    )
    for tag in META_TAGS:
        for el in soup.find_all(tag):
            # Skip metadata tags inside <ref> (citation context).
            if any(p.name == "ref" for p in el.parents if hasattr(p, "name")):
                continue
            text = _clean_text(el.get_text(" "))
            if text:
                out.append({"text": text, "doc_role": "metadata"})

    # ---- author (one row per contrib-group wrapper) ----------------------
    for el in soup.find_all("contrib-group"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "author"})

    # ---- legal (license + copyright) -------------------------------------
    for tag in ("copyright-statement", "license-p", "license"):
        for el in soup.find_all(tag):
            text = _clean_text(el.get_text(" "))
            if text:
                out.append({"text": text, "doc_role": "legal"})

    # ---- abstract -> body (relabeled) ------------------------------------
    abstract_paragraph_ids: set[int] = set()
    for el in soup.find_all("abstract"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "body"})
        for p in el.find_all("p"):
            abstract_paragraph_ids.add(id(p))

    # ---- body paragraphs --------------------------------------------------
    # Top-level <p> in <sec> / <body>. Skip <p> nested inside <ref-list>,
    # <fn>, <notes>, <abstract> (already emitted above).
    SKIP_ANCESTOR = (
        "ref-list",
        "ref",
        "fn",
        "fn-group",
        "notes",
        "abstract",
        "front",
        "back",
        "license",
        "license-p",
        "copyright-statement",
    )
    for p in soup.find_all("p"):
        if id(p) in abstract_paragraph_ids:
            continue
        skip = False
        for parent in p.parents:
            if not hasattr(parent, "name") or parent.name is None:
                continue
            if parent.name in SKIP_ANCESTOR:
                skip = True
                break
        if skip:
            continue
        text = _clean_text(p.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "body"})

    # ---- citations -------------------------------------------------------
    for el in soup.find_all("ref"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "citation"})

    # ---- footer (footnotes / notes) --------------------------------------
    for tag in ("fn", "notes"):
        for el in soup.find_all(tag):
            # <fn> inside <author-notes> in <front> still reads as footer-y
            # (corresponding-author lines, conflict-of-interest).
            text = _clean_text(el.get_text(" "))
            if text:
                out.append({"text": text, "doc_role": "footer"})

    return out


def extract_courtlistener_blocks(raw_html: str) -> list[dict]:
    """CourtListener (Free Law Project) federal opinions.

    License: Federal judicial opinions are public domain — judicial
    opinions of US courts are not copyrightable (Banks v. Manchester,
    128 U.S. 244 (1888); Wheaton v. Peters, 33 U.S. 591 (1834)). The
    Free Law Project distributes opinion text + metadata under that
    public-domain status (see https://www.courtlistener.com/help/
    api/bulk-data/). Headnotes / syllabi / publisher's editorial
    apparatus from West / Lexis are NOT included in CL bulk text — only
    the opinion-of-the-court PD text, so we accept the whole payload.
    The pair-builder must restrict to ``court__jurisdiction = F`` (federal)
    courts; state court opinions vary by state and require per-court
    license review (defer).

    CL bulk API exposes one of (in priority order):
      ``html_with_citations``  rich HTML, citations as <a> tags
      ``html``                  raw HTML
      ``html_lawbox``           CourtListener-internal cleaning
      ``html_columbia``         Columbia archive HTML
      ``plain_text``            UTF-8 text (the fetcher wraps this in
                                ``<pre>`` so the same extractor handles
                                both modes)

    Mapping (HTML mode):
      <h1>, <case-name>, .case_name, .opinion_caption  -> title
      <h2>, <h3>                                        -> body (section
                                                           headings within
                                                           the opinion are
                                                           prose context)
      <p>, <pre>                                        -> body
      <blockquote>                                      -> body (quoted
                                                           statute / prior
                                                           opinion text;
                                                           still substantive)
      <author>, <byline>, .author, .byline             -> author (per-curiam
                                                           lines, "Roberts,
                                                           C.J.", etc.)
      <citation>, .citation, <a class="citation">      -> citation
      <footnote>, .footnote, .opinion_footnote         -> footer
      <copyright>, .copyright, .license                -> legal

    For plain_text mode (no markup) the wrapper emits one <pre> block
    routed to body. Header lines like "Smith v. Jones, 123 F.3d 456 (9th
    Cir. 1997)" within the prose can't be cleanly separated and ride
    along as body — that's still domain-novel signal vs. CFR.
    """
    soup = BeautifulSoup(raw_html, "lxml")
    out: list[dict] = []

    # ---- title -----------------------------------------------------------
    seen_title_text: set[str] = set()
    title_selectors: list[tuple[str, dict]] = [
        ("h1", {}),
        ("case-name", {}),
    ]
    for tag, attrs in title_selectors:
        for el in soup.find_all(tag, attrs=attrs):
            text = _clean_text(el.get_text(" "))
            if text and text not in seen_title_text:
                out.append({"text": text, "doc_role": "title"})
                seen_title_text.add(text)
    for el in soup.find_all(class_=re.compile(r"^(case_name|opinion_caption|case-name)$", re.I)):
        text = _clean_text(el.get_text(" "))
        if text and text not in seen_title_text:
            out.append({"text": text, "doc_role": "title"})
            seen_title_text.add(text)

    # ---- author ----------------------------------------------------------
    for tag in ("author", "byline"):
        for el in soup.find_all(tag):
            text = _clean_text(el.get_text(" "))
            if text:
                out.append({"text": text, "doc_role": "author"})
    for el in soup.find_all(class_=re.compile(r"^(author|byline)$", re.I)):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "author"})

    # ---- citation --------------------------------------------------------
    citation_paragraph_ids: set[int] = set()
    for tag in ("citation",):
        for el in soup.find_all(tag):
            text = _clean_text(el.get_text(" "))
            if text:
                out.append({"text": text, "doc_role": "citation"})
                citation_paragraph_ids.add(id(el))
    for el in soup.find_all(class_=re.compile(r"^citation$", re.I)):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "citation"})
            citation_paragraph_ids.add(id(el))

    # ---- footer (footnotes) ---------------------------------------------
    footer_ids: set[int] = set()
    for tag in ("footnote",):
        for el in soup.find_all(tag):
            text = _clean_text(el.get_text(" "))
            if text:
                out.append({"text": text, "doc_role": "footer"})
                footer_ids.add(id(el))
    for el in soup.find_all(class_=re.compile(r"^(footnote|opinion_footnote)$", re.I)):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "footer"})
            footer_ids.add(id(el))

    # ---- legal -----------------------------------------------------------
    legal_ids: set[int] = set()
    for el in soup.find_all(class_=re.compile(r"^(copyright|license)$", re.I)):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "legal"})
            legal_ids.add(id(el))

    # ---- body (paragraphs / pre / blockquote / h2 / h3) -----------------
    SKIP_BODY_ANCESTOR = ("footnote", "citation", "byline", "author")
    SKIP_BODY_CLASS_RE = re.compile(
        r"^(footnote|opinion_footnote|byline|author|citation|copyright|license"
        r"|case_name|opinion_caption|case-name)$",
        re.I,
    )
    for tag in ("p", "pre", "blockquote", "h2", "h3"):
        for el in soup.find_all(tag):
            if id(el) in citation_paragraph_ids or id(el) in footer_ids or id(el) in legal_ids:
                continue
            skip = False
            for parent in el.parents:
                if not hasattr(parent, "name") or parent.name is None:
                    continue
                if parent.name in SKIP_BODY_ANCESTOR:
                    skip = True
                    break
                cls = " ".join(parent.get("class", []) or [])
                if cls and SKIP_BODY_CLASS_RE.search(cls):
                    skip = True
                    break
            if skip:
                continue
            text = _clean_text(el.get_text(" "))
            if text:
                out.append({"text": text, "doc_role": "body"})

    return out


def extract_synthetic_blocks(raw_html: str) -> list[dict]:
    """Synthetic pages (scripts/pair_from_synthetic.py) — body / quotes
    / code mixed with paragraph filler. No explicit doc-role markup."""
    soup = BeautifulSoup(raw_html, "lxml")
    out: list[dict] = []

    h1 = soup.find("h1")
    if h1:
        out.append({"text": _clean_text(h1.get_text(" ")), "doc_role": "title"})
    for el in soup.find_all("p"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "body"})
    for el in soup.find_all("blockquote"):
        text = _clean_text(el.get_text(" "))
        if text:
            out.append({"text": text, "doc_role": "body"})
    return out


def extract_forms_blocks(raw_html: str) -> list[dict]:
    """Forms — output_html IS the only HTML; doc-role signal is
    weak. Treat all text content as body for now."""
    soup = BeautifulSoup(raw_html, "lxml")
    out: list[dict] = []
    for el in soup.find_all(["p", "label", "legend", "h1", "h2"]):
        text = _clean_text(el.get_text(" "))
        if not text:
            continue
        role = "title" if el.name == "h1" else "body"
        out.append({"text": text, "doc_role": role})
    return out


SOURCE_EXTRACTORS = {
    "arxiv": extract_arxiv_blocks,
    "wikipedia": extract_wikipedia_blocks,
    "wikiquote": extract_wikipedia_blocks,
    "gutenberg": extract_gutenberg_blocks,
    "openstax": extract_openstax_blocks,
    "federal_register": extract_federal_register_blocks,
    "cfr": extract_cfr_blocks,
    "pmc": extract_pmc_blocks,
    "courtlistener": extract_courtlistener_blocks,
    "synthetic_blockquote_code": extract_synthetic_blocks,
    "pdf_form": extract_forms_blocks,
}


def extract_html_blocks(pair: dict) -> list[dict]:
    """Dispatch on pair['source']. Returns list of
    ``{text, doc_role}`` in document order."""
    source = pair.get("source", "")
    extractor = SOURCE_EXTRACTORS.get(source)
    if extractor is None:
        return []
    raw = pair.get("raw_source_html") or pair.get("raw_source_xml")
    if not raw:
        return []
    blocks = extractor(raw)
    # Apply boilerplate flag heuristic
    for b in blocks:
        b["boilerplate"] = is_boilerplate(b["text"], b["doc_role"])
    return blocks


# ---------------------------------------------------------------------------
# Per-pair worker — extract + featurize + align (NO model)
# ---------------------------------------------------------------------------
# We deliberately split the build into two phases:
#   Phase 1 (multi-process, CPU): alignment + raw row emission
#   Phase 2 (single-process, GPU): Structure-cascade attachment
#
# Reason: extract_shared + Jaccard parallelize cleanly across cores, but
# the trained Structure adapter is a single GPU resource — running it
# inside `multiprocessing` workers either deadlocks on CUDA fork or
# duplicates the model into VRAM ``workers`` times. Doing the cascade
# pass in main on already-emitted rows is also strictly cheaper than
# loading the model per pair.


# Source-name aliases — the directory on disk doesn't always match the
# slug stamped into pair["source"]. e.g. directory ``forms`` ↔ source
# ``pdf_form``. We normalize both ways so a pair from data/pairs/forms
# routes to the forms extractor regardless of which slug it carries.
_DIR_TO_SOURCE = {
    "forms": "pdf_form",
    "synthetic_blockquote_code": "synthetic_blockquote_code",
}


def _resolve_source(pair_path: Path, pair: dict) -> str:
    """Return the canonical source slug used to dispatch extractors."""
    src = pair.get("source", "") or ""
    if src in SOURCE_EXTRACTORS:
        return src
    parent = pair_path.parent.name
    if parent in _DIR_TO_SOURCE:
        return _DIR_TO_SOURCE[parent]
    if parent in SOURCE_EXTRACTORS:
        return parent
    return src


def process_pair(validator: HtmlValidator | None, work: tuple) -> dict:
    """Worker: extract+featurize one PDF, align blocks to raw_source_html
    extracted blocks via Jaccard, emit one labeled row per aligned block.

    The Structure cascade vector is left as ``None`` — Phase 2 fills it.
    """
    pair_path_str, out_examples_dir_str = work
    pair_path = Path(pair_path_str)
    out_dir = Path(out_examples_dir_str)
    stats = {
        "pair": pair_path.name,
        "aligned": 0,
        "total_blocks": 0,
        "n_html_blocks": 0,
        "source": "?",
        "error": None,
    }

    try:
        pair = json.loads(pair_path.read_text())
    except Exception as exc:
        stats["error"] = f"read pair: {exc}"
        return stats

    source = _resolve_source(pair_path, pair)
    stats["source"] = source

    html_blocks = extract_html_blocks(pair)
    stats["n_html_blocks"] = len(html_blocks)
    if not html_blocks:
        stats["error"] = "no html blocks (raw_source_html missing or empty)"
        return stats

    # Source PDF — Phase 3b convention: prefer cached extract_shared if
    # local_pdf is recorded, otherwise render output_html via Playwright.
    output_html = pair.get("output_html")
    local_pdf = pair.get("local_pdf")
    with tempfile.TemporaryDirectory() as tmp:
        if local_pdf and Path(local_pdf).exists():
            try:
                shared = extract_shared_cached(Path(local_pdf))
            except Exception as exc:
                stats["error"] = f"extract (cached): {exc}"
                return stats
        elif output_html:
            if validator is None:
                stats["error"] = "no local_pdf and no validator provided"
                return stats
            tmp_pdf = Path(tmp) / "rendered.pdf"
            try:
                validator.render_pdf(output_html, tmp_pdf)
            except Exception as exc:
                stats["error"] = f"render: {exc}"
                return stats
            try:
                shared = extract_shared(tmp_pdf)
            except Exception as exc:
                stats["error"] = f"extract: {exc}"
                return stats
        else:
            stats["error"] = "no local_pdf and no output_html"
            return stats

    examples: list[dict] = []
    html_cursor = 0
    window = 8

    # Pre-compute document-level positional totals over the SAME set of
    # pages we'll iterate (those with non-empty merged blocks). The
    # block_idx counter increments across pages × sorted blocks in the
    # same order as the alignment loop, so pos features are stable
    # regardless of how many text blocks get filtered out by jaccard.
    _pages_iter = [
        p for p in shared.get("pages", []) if (p.get("merged", {}).get("text_blocks", []) or [])
    ]
    total_pages = max(1, len(_pages_iter))
    total_blocks = max(
        1, sum(len(p.get("merged", {}).get("text_blocks", []) or []) for p in _pages_iter)
    )
    if _pages_iter:
        first_page_num = min(int(p.get("page_num", 0)) for p in _pages_iter)
    else:
        first_page_num = 0
    block_idx = 0

    for page in shared.get("pages", []):
        merged = page.get("merged", {}).get("text_blocks", []) or []
        if not merged:
            continue

        page_w = float(page.get("width", 612.0))
        page_h = float(page.get("height", 792.0))
        sizes = [b["font_size"] for b in merged if b.get("font_size") is not None]
        page_median_fs = sorted(sizes)[len(sizes) // 2] if sizes else 12.0
        heights = [
            b["bbox"][3] - b["bbox"][1]
            for b in merged
            if (b.get("bbox") and b["bbox"][3] > b["bbox"][1])
        ]
        page_median_h = sorted(heights)[len(heights) // 2] if heights else 12.0

        merged_sorted = sorted(merged, key=lambda b: (b["bbox"][1], b["bbox"][0]))

        page_num = int(page.get("page_num", 0))

        for block in merged_sorted:
            stats["total_blocks"] += 1
            # Compute positional features against ALL merged blocks (not
            # just aligned ones) so the ordinal index is stable wrt the
            # PDF layout regardless of HTML alignment yield.
            positional_vec = compute_positional_features(
                block,
                page_num=page_num,
                first_page_num=first_page_num,
                total_pages=total_pages,
                page_h=page_h,
                block_idx=block_idx,
                total_blocks=total_blocks,
            )
            block_idx += 1

            in_table = _block_in_any_table(block, page)
            text = (block.get("text") or "").strip()
            if not text:
                continue

            best_idx = -1
            best_score = 0.0
            for j in range(
                max(0, html_cursor - 2),
                min(len(html_blocks), html_cursor + window),
            ):
                s = jaccard_overlap(text, html_blocks[j]["text"])
                if s > best_score:
                    best_score = s
                    best_idx = j
            if best_idx < 0 or best_score < 0.30:
                continue

            html_meta = html_blocks[best_idx]
            doc_role = html_meta["doc_role"]
            if doc_role not in DOC_ROLE_TO_ID:
                continue
            html_cursor = max(html_cursor, best_idx + 1)

            layout_vec = compute_span_layout_features(
                block,
                page_w=page_w,
                page_h=page_h,
                page_median_fs=page_median_fs,
                page_median_h=page_median_h,
                in_table=in_table,
            )

            # Boilerplate is computed off the ALIGNED PDF text (not the
            # html block text) so the row reflects what the model will
            # actually see at inference time.
            boilerplate = is_boilerplate(text, doc_role)

            examples.append(
                {
                    "text": text,
                    "layout": layout_vec,
                    "cascade": None,  # Phase 2 fills
                    "positional": positional_vec,
                    "labels": {
                        "doc_role": DOC_ROLE_TO_ID[doc_role],
                        "boilerplate": int(boilerplate),
                    },
                    "source": source,
                    "pair": pair_path.stem,
                    "jaccard": round(best_score, 3),
                }
            )
            stats["aligned"] += 1

    if examples:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pair_path.stem}.jsonl"
        with out_path.open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return stats


# ---------------------------------------------------------------------------
# Phase 2 — Structure cascade vectors
# ---------------------------------------------------------------------------
# An 8-dim vector per row: [P(role)x6, P(is_heading=1), P(table_region=1)]
# from the trained Structure adapter. Concatenated downstream onto the
# 20-dim layout vector for a 28-dim side-channel input.

CASCADE_DIM = 6 + 1 + 1  # |role| + is_heading=1 + table_region=1


# ---------------------------------------------------------------------------
# Positional features — Semantic-only (NOT shared with Structure).
# ---------------------------------------------------------------------------
# Phase 3c precision-improvement: title and author are strongly localized to
# the first few blocks of the doc (title block 0-3, author block 1-5), but
# the 20-dim layout vec carries only page-relative position. A separate 3-dim
# positional vector exposes ordinal location in the document.
#
# IMPORTANT: positional features are appended ONLY to the Semantic
# side-channel (-> 31-dim). Structure stays on the 20-dim layout contract;
# bumping its input shape would invalidate the trained Structure adapter at
# models/council/structure/final/heads.pt (layout_norm + layout_mlp shapes).
POSITIONAL_FEATURE_DIM = 3
POSITIONAL_FEATURE_NAMES = (
    "block_index_norm",
    "page_index_norm",
    "doc_position_norm",
)


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def compute_positional_features(
    block: dict,
    *,
    page_num: int,
    first_page_num: int,
    total_pages: int,
    page_h: float,
    block_idx: int,
    total_blocks: int,
) -> list[float]:
    """3-dim positional vector capturing ordinal location in the document.

    block_index_norm    block_idx / max(1, total_blocks - 1)  in [0,1]
    page_index_norm     (page_num - first_page_num) / max(1, total_pages - 1)
    doc_position_norm   flattened y-position across all pages, in [0,1]:
        ((page_num - first_page_num) * page_h + y0) / (total_pages * page_h)
    """
    block_index_norm = block_idx / max(1, total_blocks - 1)
    page_index_norm = (page_num - first_page_num) / max(1, total_pages - 1)
    bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    y0 = float(bbox[1]) if len(bbox) >= 2 else 0.0
    doc_position_norm = ((page_num - first_page_num) * page_h + y0) / max(1.0, total_pages * page_h)
    return [
        _clip01(block_index_norm),
        _clip01(page_index_norm),
        _clip01(doc_position_norm),
    ]


def _load_structure_inference_bundle():
    """Lazy-import + load Structure adapter + tokenizer + heads. Runs
    once in main process before the cascade pass.

    Returns ``(peft_model, tokenizer, heads_dict, device)`` where
    heads_dict has keys ``role``, ``is_heading``, ``table_region``,
    ``layout_norm``, ``layout_mlp``."""
    from transformers import AutoTokenizer  # noqa: WPS433

    from dart_semantic.council import structure as bert_structure
    from dart_semantic.council.base import (
        load_lora_adapter,
        load_shared_backbone,
    )

    spec = bert_structure.ADAPTER_SPEC
    backbone = load_shared_backbone()
    adapter = load_lora_adapter(spec, backbone)

    tok_dir = spec.adapter_path / "tokenizer"
    tok = AutoTokenizer.from_pretrained(str(tok_dir) if tok_dir.exists() else backbone.name)
    heads_bundle = bert_structure._load_heads(
        spec.adapter_path / "heads.pt",
        hidden_size=backbone.hidden_size,
    )
    device = backbone.device or "cpu"
    heads = {
        "role": heads_bundle["role"].to(device),
        "is_heading": heads_bundle["is_heading"].to(device),
        "table_region": heads_bundle["table_region"].to(device),
        "layout_norm": heads_bundle["layout_norm"].to(device),
        "layout_mlp": heads_bundle["layout_mlp"].to(device),
    }
    peft_model = adapter.peft_model
    peft_model.eval()
    return peft_model, tok, heads, device


def attach_cascade_vectors(
    rows: list[dict],
    *,
    batch_size: int = 64,
) -> None:
    """Mutate ``rows`` in-place: fill ``row["cascade"]`` with the 8-dim
    Structure-derived probability vector. No-op for rows that already
    carry a cascade vector (resumability).
    """
    import torch  # noqa: WPS433

    pending = [i for i, r in enumerate(rows) if r.get("cascade") is None]
    if not pending:
        return
    peft_model, tok, heads, device = _load_structure_inference_bundle()

    head_role = heads["role"]
    head_is_heading = heads["is_heading"]
    head_table_region = heads["table_region"]
    layout_norm = heads["layout_norm"]
    layout_mlp = heads["layout_mlp"]

    n = len(pending)
    print(
        f"[cascade] running Structure adapter over {n} rows on {device} (batch_size={batch_size})",
        file=sys.stderr,
    )
    t0 = time.time()
    for start in range(0, n, batch_size):
        batch_idx = pending[start : start + batch_size]
        batch_texts = [rows[i]["text"] for i in batch_idx]
        batch_layouts = [rows[i]["layout"] for i in batch_idx]
        enc = tok(
            batch_texts, padding=True, truncation=True, max_length=192, return_tensors="pt"
        ).to(device)
        layout_t = torch.tensor(batch_layouts, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = peft_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            pooled = out.last_hidden_state[:, 0, :].float()
            layout_h = layout_mlp(layout_norm(layout_t))
            h = torch.cat([pooled, layout_h], dim=-1)
            p_role = torch.softmax(head_role(h), dim=-1)
            p_is_heading = torch.softmax(head_is_heading(h), dim=-1)
            p_table = torch.softmax(head_table_region(h), dim=-1)
        for k, row_idx in enumerate(batch_idx):
            vec = (
                p_role[k].tolist()
                + [float(p_is_heading[k][1].item())]
                + [float(p_table[k][1].item())]
            )
            rows[row_idx]["cascade"] = vec
        if (start // batch_size) % 50 == 0:
            print(
                f"  [cascade] {start + len(batch_idx)}/{n}  ({(time.time() - t0):.1f}s)",
                file=sys.stderr,
            )
    print(f"[cascade] done in {(time.time() - t0):.1f}s", file=sys.stderr)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split(
    rows: list[dict],
    *,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratify by doc_role so rare classes appear in all splits."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["labels"]["doc_role"]].append(r)
    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for _, lst in by_class.items():
        rng.shuffle(lst)
        n = len(lst)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(lst[:n_train])
        val.extend(lst[n_train : n_train + n_val])
        test.extend(lst[n_train + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_DEFAULT_PAIR_DIRS = [
    Path("data/pairs/arxiv"),
    Path("data/pairs/wikipedia"),
    Path("data/pairs/openstax"),
    Path("data/pairs/federal_register"),
    Path("data/pairs/cfr"),
    Path("data/pairs/pmc"),
    Path("data/pairs/courtlistener"),
    Path("data/pairs/gutenberg"),
    Path("data/pairs/forms"),
    Path("data/pairs/synthetic_blockquote_code"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Sample a few pairs per source and print extractor counts for spot-check.",
    )
    ap.add_argument("--n-per-source", type=int, default=2)
    ap.add_argument("--pair-dirs", type=Path, nargs="+", default=_DEFAULT_PAIR_DIRS)
    ap.add_argument("--examples-dir", type=Path, default=Path("data/semantic_dataset/per_pair"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/semantic_dataset"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-source", type=int, default=None)
    add_cap_args(ap)
    ap.add_argument(
        "--skip-cascade",
        action="store_true",
        help="Skip Phase 2 cascade attachment (debug; produces dataset with cascade=None).",
    )
    ap.add_argument("--cascade-batch-size", type=int, default=64)
    args = ap.parse_args()

    if args.smoke:
        for source in SOURCE_EXTRACTORS:
            # forms lives in data/pairs/forms but its source slug is pdf_form.
            pair_dir = (
                Path("data/pairs/forms") if source == "pdf_form" else Path(f"data/pairs/{source}")
            )
            files = sorted(pair_dir.glob("*.json"))[: args.n_per_source]
            print(f"\n=== {source} ({len(files)} pair files) ===")
            for f in files:
                pair = json.loads(f.read_text())
                blocks = extract_html_blocks(pair)
                role_counts = Counter(b["doc_role"] for b in blocks)
                bp = sum(b.get("boilerplate", 0) for b in blocks)
                print(f"  {f.name:60s}  {len(blocks)} blocks  bp={bp}  {dict(role_counts)}")
        return

    # ---- Phase 1: alignment ------------------------------------------------
    pair_paths: list[Path] = []
    for d in args.pair_dirs:
        if not d.exists():
            continue
        files = sorted(d.glob("*.json"))
        if args.limit_per_source:
            files = files[: args.limit_per_source]
        pair_paths.extend(files)
    print(f"[plan] {len(pair_paths)} pair files across {len(args.pair_dirs)} dirs", file=sys.stderr)

    existing = (
        {p.stem for p in args.examples_dir.glob("*.jsonl")} if args.examples_dir.exists() else set()
    )
    pair_paths = [p for p in pair_paths if p.stem not in existing]
    print(f"[plan] {len(pair_paths)} to process ({len(existing)} done)", file=sys.stderr)

    work_items = [(str(p), str(args.examples_dir)) for p in pair_paths]
    start = time.time()
    done = 0
    totals = {"aligned": 0, "total_blocks": 0, "errors": 0, "n_html_blocks": 0}
    err_by_source: Counter = Counter()
    for stats in run_in_pool(process_pair, work_items, workers=args.workers):
        done += 1
        if stats.get("error"):
            totals["errors"] += 1
            err_by_source[stats.get("source", "?")] += 1
            if done % 25 == 1:
                print(
                    f"[{done}/{len(work_items)}] {stats['pair'][:50]} ERR {stats['error'][:60]}",
                    file=sys.stderr,
                )
        else:
            totals["aligned"] += stats["aligned"]
            totals["total_blocks"] += stats["total_blocks"]
            totals["n_html_blocks"] += stats["n_html_blocks"]
            if done % 50 == 0 or done == len(work_items):
                rate = totals["aligned"] / max(1, totals["total_blocks"]) * 100
                print(
                    f"[{done}/{len(work_items)}] aligned="
                    f"{totals['aligned']}  align_rate={rate:.1f}%  "
                    f"elapsed={(time.time() - start) / 60:.1f}min",
                    file=sys.stderr,
                )
    if err_by_source:
        print(f"[errors by source] {dict(err_by_source)}", file=sys.stderr)

    # ---- Merge per-pair JSONL ---------------------------------------------
    all_examples: list[dict] = []
    for p in sorted(args.examples_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                all_examples.append(json.loads(line))
    print(f"\n[merge] total labeled examples: {len(all_examples)}", file=sys.stderr)
    if not all_examples:
        raise SystemExit("no examples produced")

    # ---- Balance: cap dominant sources before the expensive cascade pass ---
    all_examples, cap_report = apply_caps_and_report(all_examples, args, label_key="doc_role")

    # ---- Phase 2: cascade attachment --------------------------------------
    if args.skip_cascade:
        print("[cascade] skipped (--skip-cascade)", file=sys.stderr)
    else:
        attach_cascade_vectors(
            all_examples,
            batch_size=args.cascade_batch_size,
        )

    # ---- Stratified split + write -----------------------------------------
    train, val, test = stratified_split(all_examples, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[write] {path}  {len(rows)} rows", file=sys.stderr)

    # ---- Coverage report --------------------------------------------------
    role_counts = Counter(r["labels"]["doc_role"] for r in all_examples)
    bp_counts = Counter(r["labels"]["boilerplate"] for r in all_examples)
    source_counts = Counter(r["source"] for r in all_examples)

    print("\n[balance] doc_role:")
    for cls_id, n in role_counts.most_common():
        pct = 100.0 * n / max(1, len(all_examples))
        print(f"  {DOC_ROLE_NAMES[cls_id]:24} {n:7}  ({pct:5.2f}%)")
    print("\n[balance] boilerplate:")
    for k in (0, 1):
        n = bp_counts.get(k, 0)
        pct = 100.0 * n / max(1, len(all_examples))
        label = "boilerplate" if k else "not_boilerplate"
        print(f"  {label:24} {n:7}  ({pct:5.2f}%)")
    print("\n[balance] by source:")
    for src, n in source_counts.most_common():
        print(f"  {src:24} {n}")

    coverage = {
        "n_examples_total": len(all_examples),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "doc_role_counts": {DOC_ROLE_NAMES[k]: v for k, v in role_counts.items()},
        "boilerplate_counts": {str(k): v for k, v in bp_counts.items()},
        "source_counts": dict(source_counts),
        "source_caps": cap_report,
        "max_examples_per_source": args.max_examples_per_source,
        "errors_by_source": dict(err_by_source),
        "layout_feature_dim": LAYOUT_FEATURE_DIM,
        "layout_feature_names": LAYOUT_FEATURE_NAMES,
        "cascade_dim": CASCADE_DIM,
        "positional_dim": POSITIONAL_FEATURE_DIM,
        "positional_feature_names": list(POSITIONAL_FEATURE_NAMES),
        "doc_role_names": list(DOC_ROLE_NAMES),
        "cascade_skipped": bool(args.skip_cascade),
    }
    (args.out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2))
    print(f"[save] coverage report -> {args.out_dir / 'coverage_report.json'}")


if __name__ == "__main__":
    main()

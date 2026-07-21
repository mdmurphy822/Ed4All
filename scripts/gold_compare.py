#!/usr/bin/env python3
"""Gold-comparison QA harness for SemantiK PDF conversions and their chunksets.

Measures a *candidate* textbook conversion (SemantiK accessible-HTML output
and/or the pipeline chunking phase's ``chunks.jsonl``) against a known-good
*gold* ``chunk_v4`` chunkset of the *same* textbook, and flags conversion
defects. Designed to be run repeatedly mid-flight during a long conversion run,
once per completed chapter.

The gold set is assumed to be built from clean vendor text, so its TEXT is the
reference for what an OCR-driven conversion should recover.

Gold schema fields consumed (introspected from a real ``chunk_v4`` file, not
guessed):
  * ``text``                              -- reference text of the chunk
  * ``source.item_path``                  -- ``...chNN...`` -> chapter number
  * ``source.source_references[].sourceId`` -- ``semantik:...#<section-slug>``; the
        fragment after ``#`` is the canonical SECTION key (e.g.
        ``1-1-introduction-to-whole-numbers``). ~7-15 sections/chapter.
  * ``source.section_heading``            -- human sub-heading (fallback only)

Candidate inputs (either or both):
  (a) --candidate-html DIR  : SemantiK accessible-HTML, one file per chapter PDF
        (chapter derived from a ``chNN`` substring in the filename, or pinned
        explicitly with ``--map chNN=path``). HTML is stripped to text.
  (b) --candidate-chunks FILE : a chunk_v4 chunks.jsonl produced by the chunking
        phase over the converted corpus (chapter derived from each candidate
        chunk's own item_path / sourceId / tags, with a best-text-match
        fallback).

Metrics per chapter + overall (all deterministic, CPU-only, stdlib + sklearn/
numpy if present -- NO embedding models are loaded):
  1. TEXT RECALL   -- 5-gram shingle containment of each gold chunk in the
        candidate chapter text (distribution + mean; worst-10 gold chunks).
  2. JUNK RATE / PRECISION -- OOV-token rate: candidate content tokens absent
        from the gold chapter vocabulary (OCR garbage / hallucinated
        boilerplate). Plus an informational 3-gram ``novel_shingle_rate``.
  3. MISHANDLED-TEXT -- mojibake/encoding artifacts, OCR word-shape garbage,
        symbol density, broken line-break hyphenation, ALL-CAPS run anomalies.
  4. REPETITION    -- excess duplicated 12-grams (beyond gold frequency),
        repeated consecutive sentences, and (chunks mode) near-duplicate
        candidate chunks (Jaccard > 0.85).
  5. CHUNK BLEEDING (chunks mode) -- candidate chunks whose text spans 2+ gold
        SECTIONS, plus boundary sentence duplication across adjacent chunks.
  6. STRUCTURE sanity (html mode) -- heading count + level histogram vs the
        gold per-chapter section list; missing / extra section headings named.
  7. Per-chapter one-line verdict PASS / WARN / FAIL with triggering detectors.

Verdict thresholds (documented here = single source of truth):
  FAIL if any of:
     recall_mean           < FAIL_RECALL        (0.70)
     junk_rate             > FAIL_JUNK          (0.25)
     mojibake_per_1k_chars > FAIL_MOJIBAKE      (2.0)
     high-severity bleeds  > 0   (chunks mode)
     chapter absent from candidate entirely
  WARN if any of:
     recall_mean           < WARN_RECALL        (0.85)
     junk_rate             > WARN_JUNK          (0.10)
     garbage_word_rate     > WARN_GARBAGE       (0.05)
     broken_hyphen_per_1k  > WARN_HYPHEN        (1.0)
     any bleed             > 0   (chunks mode)
     repeated_12gram_excess> WARN_REPEAT_EXCESS (3)
     near_dup_chunk_pairs  > 0   (chunks mode)
     missing_sections      > WARN_MISSING_SECT  (1)  (html mode)
  PASS otherwise.

Usage:
  .venv/bin/python scripts/gold_compare.py \\
      --gold <gold_chunks.jsonl> \\
      [--candidate-html DIR | --candidate-chunks FILE] \\
      [--map ch01=/path/a.html --map ch02=...] \\
      [--chapters 1,2,3] \\
      --json-out report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

# Shared math-representation folding (LaTeX ↔ unicode ↔ plain). Lives in
# lib/semantik/ so both this harness and the SemantiK cascade share one
# implementation (mirrors vlm_fusion._strip_latex — see that module's docstring).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.semantik.math_fold import (  # noqa: E402
    MATH_WORDS,
    count_math_folds,
    fold_math,
)

SCHEMA_VERSION = "gold_compare/1.1"

# ---- tunable constants (documented in the module docstring) ---------------
SHINGLE_N = 5           # word n-gram for recall containment
PRECISION_N = 3         # word n-gram for informational novel-shingle rate
REPEAT_N = 12           # word n-gram for repetition detector
NEAR_DUP_JACCARD = 0.85 # candidate-chunk near-duplicate threshold
MIN_SENT_WORDS = 8      # sentence length floor for repetition / boundary checks

# bleed detection
BLEED_MIN_SHINGLES = 5          # a section needs >= this many DISCRIMINATIVE shingles to count
BLEED_HIGH_RATIO = 0.50         # 2nd-section overlap / 1st >= this => high severity
# A gold section holding more than this fraction of a chapter's chunks is a
# "lumping" residue bucket (mis-anchored chunks pooled onto one section anchor).
# Shingles it *solely* owns are excluded from the discriminative map so the bleed
# detector can't fire on a mega-section artifact. Only enforced once a chapter has
# at least OVERSIZED_SECTION_MIN_CHUNKS chunks, so a legitimately single-section
# small chapter isn't gutted.
OVERSIZED_SECTION_FRACTION = 0.25
OVERSIZED_SECTION_MIN_CHUNKS = 4

# verdict thresholds
FAIL_RECALL = 0.70
FAIL_JUNK = 0.25
FAIL_MOJIBAKE = 2.0
WARN_RECALL = 0.85
WARN_JUNK = 0.10
WARN_GARBAGE = 0.05
WARN_HYPHEN = 1.0
WARN_REPEAT_EXCESS = 3
WARN_MISSING_SECT = 1

_WORD_RE = re.compile(r"[a-z0-9]+")
_MOJIBAKE_RE = re.compile(r"Ã.|â€|Â[ ]|Â\b|�|â€™|â€œ|â€\x9d|Ã©|Ã¨|â€“|â€”")
_HYPHEN_BREAK_RE = re.compile(r"[A-Za-z]{2,}-\s+[a-z]{2,}")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CHAP_RE = re.compile(r"ch0*(\d+)", re.IGNORECASE)

_STOPWORDS = frozenset(
    "the a an of to and or in on at for is are be as by with that this it we you "
    "our your they he she his her its from into out up down not no yes if then "
    "than so but can will may these those which who whom whose have has had do "
    "does did each any all more most some such only just also very".split()
)


# --------------------------------------------------------------------------
# tokenization / shingling helpers
# --------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens."""
    return _WORD_RE.findall(text.lower())


def fold_tokenize(text: str) -> list[str]:
    """Tokenize AFTER math-representation folding (LaTeX ↔ unicode ↔ plain).

    Used everywhere candidate text is scored against the gold vocabulary /
    shingles (recall, junk, garbage, repetition, bleed) so both sides meet in
    one canonical math vocabulary. A strict no-op on plain English, so
    non-math corpora tokenize byte-identically to :func:`tokenize`.
    """
    return _WORD_RE.findall(fold_math(text).lower())


def ngram_set(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_counter(tokens: list[str], n: int) -> Counter:
    c: Counter = Counter()
    if len(tokens) < n:
        return c
    for i in range(len(tokens) - n + 1):
        c[tuple(tokens[i : i + n])] += 1
    return c


def sentences(text: str) -> list[str]:
    out = []
    for raw in _SENT_SPLIT_RE.split(text):
        s = re.sub(r"\s+", " ", raw).strip()
        if s:
            out.append(s)
    return out


def normalize_sentence(s: str) -> tuple[str, ...]:
    return tuple(tokenize(s))


# --------------------------------------------------------------------------
# HTML stripping (+ heading capture)
# --------------------------------------------------------------------------
class _HTMLTextExtractor(HTMLParser):
    _SKIP = {"script", "style", "head", "noscript"}
    _HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    # Screen-reader-only / hidden a11y elements. SemantiK's gold-shell emits
    # screen-reader-only structural labels — e.g.
    # ``<p class="sr-only" hidden>Paragraph block</p>`` — as an accessibility
    # surface. Those labels are NOT document content: left in, thousands of
    # "Paragraph block" / "List block" tokens inflate the junk rate and skew
    # recall / repetition. Any element carrying a screen-reader-only class, a
    # bare ``hidden`` attribute, or ``aria-hidden="true"`` is dropped, subtree
    # and all, before the text metrics run.
    _HIDDEN_CLASSES = {
        "sr-only",
        "visually-hidden",
        "visuallyhidden",
        "screen-reader-only",
        "screen-reader-text",
    }
    # Void elements never carry an end tag, so they must never push onto the
    # open-element stack (doing so would strand a stale skip region).
    _VOID = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        # open-element stack of [tag, is_hidden]; ``_skip_depth`` mirrors the
        # count of currently-open hidden/skipped ancestors so text under any of
        # them is discarded.
        self._open: list[list] = []
        self._skip_depth = 0
        self._heading_stack: list[list] = []  # [level, [text parts]]
        self.headings: list[tuple[int, str]] = []

    @classmethod
    def _is_hidden(cls, tag, attrs) -> bool:
        if tag in cls._SKIP:
            return True
        for name, value in attrs:
            lname = name.lower()
            if lname == "hidden":
                return True
            if lname == "aria-hidden" and (value or "").strip().lower() == "true":
                return True
            if lname == "class" and value:
                if any(c in cls._HIDDEN_CLASSES for c in value.lower().split()):
                    return True
        return False

    def handle_starttag(self, tag, attrs):
        hidden = self._is_hidden(tag, attrs)
        if tag not in self._VOID:
            self._open.append([tag, hidden])
            if hidden:
                self._skip_depth += 1
        # Only open a heading capture when NOT inside a hidden/skipped region.
        if tag in self._HEADINGS and self._skip_depth == 0:
            self._heading_stack.append([int(tag[1]), []])

    def handle_endtag(self, tag):
        # Unwind the open-element stack back to the matching start tag,
        # tolerating malformed / unbalanced nesting.
        for i in range(len(self._open) - 1, -1, -1):
            if self._open[i][0] == tag:
                for _t, hid in self._open[i:]:
                    if hid:
                        self._skip_depth -= 1
                del self._open[i:]
                break
        if tag in self._HEADINGS and self._heading_stack:
            level, parts = self._heading_stack.pop()
            htext = re.sub(r"\s+", " ", "".join(parts)).strip()
            if htext:
                self.headings.append((level, htext))
        # block-level tags -> whitespace so words don't fuse
        self._chunks.append(" ")

    def handle_data(self, data):
        if self._skip_depth:
            return
        self._chunks.append(data)
        if self._heading_stack:
            self._heading_stack[-1][1].append(data)

    def text(self) -> str:
        return re.sub(r"[ \t]+", " ", "".join(self._chunks))


def strip_html(html: str) -> tuple[str, list[tuple[int, str]]]:
    p = _HTMLTextExtractor()
    try:
        p.feed(html)
        p.close()
    except Exception:
        # extremely malformed HTML -> best-effort tag strip
        return re.sub(r"<[^>]+>", " ", html), []
    return p.text(), p.headings


# --------------------------------------------------------------------------
# chapter attribution
# --------------------------------------------------------------------------
def chapter_from_text(*candidates: str | None) -> int | None:
    for c in candidates:
        if not c:
            continue
        m = _CHAP_RE.search(c)
        if m:
            return int(m.group(1))
    return None


# --------------------------------------------------------------------------
# gold model
# --------------------------------------------------------------------------
@dataclass
class GoldChunk:
    id: str
    text: str
    chapter: int | None
    section: str  # section slug (sourceId fragment)
    heading: str


@dataclass
class GoldChapter:
    chapter: int
    chunks: list[GoldChunk] = field(default_factory=list)
    # lazily built caches
    vocab: set[str] = field(default_factory=set)
    shingles: set[tuple[str, ...]] = field(default_factory=set)
    tri_shingles: set[tuple[str, ...]] = field(default_factory=set)
    repeat_counts: Counter = field(default_factory=Counter)
    section_shingles: dict[str, set[tuple[str, ...]]] = field(default_factory=dict)
    # 5-gram -> the single owning section, ONLY for shingles unique to one
    # section (discriminative). Shared boilerplate shingles are excluded so the
    # bleed detector never fires on legitimately-repeated instructional phrasing.
    discriminative: dict[tuple[str, ...], str] = field(default_factory=dict)
    section_titles: dict[str, str] = field(default_factory=dict)


def _primary_source_id(source: dict) -> str | None:
    refs = source.get("source_references") or []
    for r in refs:
        if r.get("role") == "primary" and r.get("sourceId"):
            return r["sourceId"]
    for r in refs:
        if r.get("sourceId"):
            return r["sourceId"]
    return None


def section_slug_from_source(source: dict) -> str:
    sid = _primary_source_id(source)
    if sid and "#" in sid:
        return sid.split("#", 1)[1]
    if sid:
        return sid
    # fallback: section_heading slugified
    sh = source.get("section_heading") or "unknown"
    return re.sub(r"[^a-z0-9]+", "-", sh.lower()).strip("-") or "unknown"


def _fragment_of(sid: str) -> str:
    """Section-slug fragment of a sourceId (part after '#', else the whole id)."""
    if "#" in sid:
        return sid.split("#", 1)[1]
    return sid


def majority_section_slug_from_source(source: dict) -> str:
    """Derive a gold chunk's section from the MAJORITY section anchor across ALL
    of its ``source.source_references[]`` sourceId fragments.

    A gold chunk carries multiple source_references; anchoring it on only the
    primary-or-first ref (``section_slug_from_source``) lumps 40-50% of a
    chapter's chunks onto the chapter's first section anchor and poisons the
    discriminative-shingle map. Here we take the fragment (part after ``#``, else
    the whole sourceId) that occurs most often across the whole ref list. Ties
    break to the FIRST ref's fragment (document order -> deterministic). With no
    usable sourceIds, fall back to the ``section_heading`` slug (same as
    ``section_slug_from_source``).
    """
    refs = source.get("source_references") or []
    fragments: list[str] = [
        _fragment_of(r["sourceId"]) for r in refs if r.get("sourceId")
    ]
    if not fragments:
        sh = source.get("section_heading") or "unknown"
        return re.sub(r"[^a-z0-9]+", "-", sh.lower()).strip("-") or "unknown"
    counts = Counter(fragments)
    top = max(counts.values())
    # tie-break: earliest fragment (document order) among those at the max count
    for frag in fragments:
        if counts[frag] == top:
            return frag
    return fragments[0]  # unreachable; defensive


def section_title(slug: str) -> str:
    """Human-ish title from a section slug: drop leading numeric prefix."""
    body = re.sub(r"^\d+(-\d+)?-", "", slug)
    return body.replace("-", " ").strip()


def load_gold(path: Path) -> dict[int, GoldChapter]:
    chapters: dict[int, GoldChapter] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            src = obj.get("source", {}) or {}
            ch = chapter_from_text(
                src.get("item_path"),
                src.get("module_id"),
                _primary_source_id(src),
            )
            slug = majority_section_slug_from_source(src)
            gc = GoldChunk(
                id=obj.get("id", ""),
                text=obj.get("text", "") or "",
                chapter=ch,
                section=slug,
                heading=src.get("section_heading", "") or "",
            )
            chapters.setdefault(ch, GoldChapter(chapter=ch)).chunks.append(gc)
    for gch in chapters.values():
        _build_gold_caches(gch)
    return chapters


def _build_gold_caches(gch: GoldChapter) -> None:
    all_tokens: list[str] = []
    sec_tokens: dict[str, list[str]] = defaultdict(list)
    for c in gch.chunks:
        toks = fold_tokenize(c.text)
        all_tokens.extend(toks)
        sec_tokens[c.section].extend(toks)
        gch.section_titles.setdefault(c.section, section_title(c.section))
    gch.vocab = set(all_tokens)
    gch.shingles = ngram_set(all_tokens, SHINGLE_N)
    gch.tri_shingles = ngram_set(all_tokens, PRECISION_N)
    gch.repeat_counts = ngram_counter(all_tokens, REPEAT_N)
    for slug, toks in sec_tokens.items():
        gch.section_shingles[slug] = ngram_set(toks, SHINGLE_N)
    # Defense in depth against lumping residue: a section bucket holding more
    # than OVERSIZED_SECTION_FRACTION of the chapter's chunks is suspect (mis-
    # anchored chunks pooled onto one anchor). Exclude shingles it SOLELY owns
    # from the discriminative map so the bleed detector can't fire on a lumped
    # mega-section's shingles. Only enforced once the chapter is big enough that
    # an oversized bucket is a genuine anomaly (not a small single-section
    # chapter).
    n_chunks = len(gch.chunks)
    oversized: set[str] = set()
    if n_chunks >= OVERSIZED_SECTION_MIN_CHUNKS:
        sec_chunk_counts: Counter = Counter(c.section for c in gch.chunks)
        threshold = OVERSIZED_SECTION_FRACTION * n_chunks
        oversized = {slug for slug, cnt in sec_chunk_counts.items() if cnt > threshold}
    # a shingle is discriminative iff it lives in exactly one gold section
    shingle_owners: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for slug, shset in gch.section_shingles.items():
        for sh in shset:
            shingle_owners[sh].add(slug)
    gch.discriminative = {
        sh: next(iter(owners))
        for sh, owners in shingle_owners.items()
        if len(owners) == 1 and next(iter(owners)) not in oversized
    }


# --------------------------------------------------------------------------
# candidate model
# --------------------------------------------------------------------------
@dataclass
class CandChunk:
    id: str
    text: str
    chapter: int | None
    fallback_assigned: bool = False


@dataclass
class CandChapter:
    chapter: int
    text: str = ""
    headings: list[tuple[int, str]] = field(default_factory=list)
    chunks: list[CandChunk] = field(default_factory=list)  # chunks mode only
    fallback_chunks: int = 0


def load_candidate_html(
    html_dir: Path, explicit_map: dict[int, Path]
) -> dict[int, CandChapter]:
    out: dict[int, CandChapter] = {}
    files: dict[int, Path] = dict(explicit_map)
    if html_dir is not None:
        for p in sorted(html_dir.glob("*.htm*")):
            ch = chapter_from_text(p.name)
            if ch is not None and ch not in files:
                files[ch] = p
    for ch, p in files.items():
        text, headings = strip_html(p.read_text(encoding="utf-8", errors="replace"))
        out[ch] = CandChapter(chapter=ch, text=text, headings=headings)
    return out


def load_candidate_chunks(
    chunks_path: Path, gold: dict[int, GoldChapter]
) -> dict[int, CandChapter]:
    raw: list[CandChunk] = []
    unassigned: list[CandChunk] = []
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            src = obj.get("source", {}) or {}
            ch = chapter_from_text(
                src.get("item_path"),
                src.get("module_id"),
                _primary_source_id(src),
                *(obj.get("concept_tags") or []),
                obj.get("id", ""),
            )
            cc = CandChunk(id=obj.get("id", ""), text=obj.get("text", "") or "", chapter=ch)
            raw.append(cc)
            if ch is None:
                unassigned.append(cc)

    # best-text-match fallback for chunks with no chNN signal anywhere
    if unassigned:
        gold_sh = {ch: gch.shingles for ch, gch in gold.items() if ch is not None}
        for cc in unassigned:
            toks = fold_tokenize(cc.text)
            csh = ngram_set(toks, SHINGLE_N)
            best_ch, best_ov = None, 0
            if csh:
                for ch, gsh in gold_sh.items():
                    ov = len(csh & gsh)
                    if ov > best_ov:
                        best_ch, best_ov = ch, ov
            cc.chapter = best_ch
            cc.fallback_assigned = True

    out: dict[int, CandChapter] = {}
    for cc in raw:
        cch = out.setdefault(cc.chapter, CandChapter(chapter=cc.chapter))
        cch.chunks.append(cc)
        if cc.fallback_assigned:
            cch.fallback_chunks += 1
    for cch in out.values():
        cch.text = "\n\n".join(c.text for c in cch.chunks)
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def metric_recall(gch: GoldChapter, cand_text: str) -> dict:
    cand_tokens = fold_tokenize(cand_text)
    cand_shingles = ngram_set(cand_tokens, SHINGLE_N)
    cand_vocab = set(cand_tokens)
    per_chunk = []
    for c in gch.chunks:
        gtoks = fold_tokenize(c.text)
        gsh = ngram_set(gtoks, SHINGLE_N)
        if gsh:
            rec = len(gsh & cand_shingles) / len(gsh)
        elif gtoks:  # too short for a 5-gram -> token-set containment
            gset = set(gtoks)
            rec = len(gset & cand_vocab) / len(gset)
        else:
            rec = 1.0
        per_chunk.append((rec, c))
    recalls = [r for r, _ in per_chunk]
    mean = sum(recalls) / len(recalls) if recalls else 0.0
    worst = sorted(per_chunk, key=lambda t: t[0])[:10]
    return {
        "gold_chunks": len(recalls),
        "recall_mean": round(mean, 4),
        "recall_min": round(min(recalls), 4) if recalls else 0.0,
        "recall_p10": round(_pct(recalls, 0.10), 4),
        "recall_p50": round(_pct(recalls, 0.50), 4),
        "fraction_below_0.5": round(sum(1 for r in recalls if r < 0.5) / len(recalls), 4)
        if recalls
        else 0.0,
        "worst_10": [
            {
                "gold_id": c.id,
                "recall": round(r, 4),
                "section": c.section,
                "heading": c.heading,
                "preview": re.sub(r"\s+", " ", c.text)[:90],
            }
            for r, c in worst
        ],
    }


def metric_junk(gch: GoldChapter, cand_text: str) -> dict:
    cand_tokens = fold_tokenize(cand_text)
    content = [t for t in cand_tokens if len(t) >= 2 and not t.isdigit()]
    # Folded LaTeX command words (``sqrt``, ``frac``, ``cdot`` …) are legitimate
    # math tokens once fold_math has run — never junk. Exempt them from the OOV
    # set exactly as ``_looks_like_garbage`` does, so a math-heavy chapter whose
    # GOLD reference happened to express no such notation (e.g. gold ch09 dropped
    # every radical) is not scored down for CORRECTLY emitting it.
    oov = [t for t in content if t not in gch.vocab and t not in MATH_WORDS]
    junk_rate = len(oov) / len(content) if content else 0.0
    cand_tri = ngram_set(cand_tokens, PRECISION_N)
    novel = cand_tri - gch.tri_shingles
    novel_rate = len(novel) / len(cand_tri) if cand_tri else 0.0
    oov_counter = Counter(oov)
    return {
        "candidate_content_tokens": len(content),
        "junk_rate": round(junk_rate, 4),
        "precision": round(1.0 - junk_rate, 4),
        "novel_shingle_rate": round(novel_rate, 4),
        "top_oov_tokens": [
            {"token": t, "count": n} for t, n in oov_counter.most_common(15)
        ],
    }


def _looks_like_garbage(tok: str) -> bool:
    if len(tok) < 4 or any(ch.isdigit() for ch in tok):
        return False
    # Folded LaTeX command words (``sqrt``, ``frac``, ``cdot`` …) are legitimate
    # math tokens once fold_math has run — never OCR garble. Exempt them BEFORE
    # the no-vowel rule (``sqrt`` / ``frac`` would otherwise trip it).
    if tok in MATH_WORDS:
        return False
    if not any(v in tok for v in "aeiou"):
        return True
    # 6+ consonant run (raised from 5): English has legitimate 5-consonant
    # clusters ("offspring" → "ffspr", "strengths" → "ngths"); a 6+ run with a
    # vowel is still garble. True vowel-less garble is caught by the rule above.
    if re.search(r"[bcdfghjklmnpqrstvwxz]{6,}", tok):
        return True
    if re.search(r"(.)\1\1", tok):  # 3+ repeated same char
        return True
    return False


def metric_mishandled(cand_text: str) -> dict:
    n_chars = max(1, len(cand_text))
    words = tokenize(cand_text)
    n_words = max(1, len(words))
    # Garbage-word detection runs on the FOLDED token stream so LaTeX control
    # words (``sqrt`` × thousands) and digit-letter fusions are normalized away
    # before the word-shape heuristic — they are representation, not garble.
    folded_words = fold_tokenize(cand_text)
    n_folded = max(1, len(folded_words))
    mojibake = len(_MOJIBAKE_RE.findall(cand_text))
    replacement = cand_text.count("�")
    hyphen_breaks = len(_HYPHEN_BREAK_RE.findall(cand_text))
    non_space = [c for c in cand_text if not c.isspace()]
    symbols = sum(1 for c in non_space if not (c.isalnum() or c in ".,;:!?'\"()-[]%/$#&"))
    symbol_density = symbols / max(1, len(non_space))
    garbage_words = sum(1 for w in folded_words if _looks_like_garbage(w))
    # ALL-CAPS runs: >=4 consecutive uppercase words (len>=2) in the raw text
    caps_tokens = re.findall(r"[A-Za-z]{2,}", cand_text)
    allcaps_runs = 0
    run = 0
    for t in caps_tokens:
        if t.isupper():
            run += 1
            if run == 4:
                allcaps_runs += 1
        else:
            run = 0
    return {
        "mojibake_hits": mojibake,
        "replacement_chars": replacement,
        "mojibake_per_1k_chars": round(1000.0 * (mojibake + replacement) / n_chars, 4),
        "broken_hyphen_hits": hyphen_breaks,
        "broken_hyphen_per_1k_words": round(1000.0 * hyphen_breaks / n_words, 4),
        "symbol_density": round(symbol_density, 4),
        "garbage_words": garbage_words,
        "garbage_word_rate": round(garbage_words / n_folded, 4),
        "allcaps_runs": allcaps_runs,
    }


def metric_repetition(gch: GoldChapter, cand_text: str, cand_chunks: list[CandChunk]) -> dict:
    cand_tokens = fold_tokenize(cand_text)
    cand_12 = ngram_counter(cand_tokens, REPEAT_N)
    excess = []
    total_excess = 0
    for gram, ccount in cand_12.items():
        gcount = gch.repeat_counts.get(gram, 0)
        ex = ccount - gcount
        if ccount > 3 and ex > 3:
            excess.append((ex, gram))
            total_excess += ex
    excess.sort(reverse=True)
    # repeated consecutive sentences
    sents = [normalize_sentence(s) for s in sentences(cand_text)]
    sents = [s for s in sents if len(s) >= MIN_SENT_WORDS]
    consec_dupes = sum(1 for i in range(1, len(sents)) if sents[i] == sents[i - 1])
    # near-duplicate candidate chunks (chunks mode)
    near_dup_pairs = 0
    near_dup_examples = []
    if cand_chunks:
        tok_sets = [(c.id, set(fold_tokenize(c.text))) for c in cand_chunks]
        tok_sets = [(cid, s) for cid, s in tok_sets if len(s) >= 5]
        for i in range(len(tok_sets)):
            id_i, s_i = tok_sets[i]
            for j in range(i + 1, len(tok_sets)):
                id_j, s_j = tok_sets[j]
                inter = len(s_i & s_j)
                if inter == 0:
                    continue
                union = len(s_i | s_j)
                if union and inter / union > NEAR_DUP_JACCARD:
                    near_dup_pairs += 1
                    if len(near_dup_examples) < 10:
                        near_dup_examples.append(
                            {"a": id_i, "b": id_j, "jaccard": round(inter / union, 3)}
                        )
    return {
        "repeated_12gram_blocks": len(excess),
        "repeated_12gram_excess": total_excess,
        "top_repeated_12grams": [
            {"excess": ex, "ngram": " ".join(g)} for ex, g in excess[:8]
        ],
        "consecutive_duplicate_sentences": consec_dupes,
        "near_duplicate_chunk_pairs": near_dup_pairs,
        "near_duplicate_examples": near_dup_examples,
    }


def metric_bleed(gch: GoldChapter, cand_chunks: list[CandChunk]) -> dict:
    bleeds = []
    high = 0
    disc = gch.discriminative  # 5-gram -> owning section (unique-to-one-section only)
    for c in cand_chunks:
        toks = fold_tokenize(c.text)
        csh = ngram_set(toks, SHINGLE_N)
        if len(csh) < BLEED_MIN_SHINGLES:
            continue
        # count only *discriminative* shingles per section (boilerplate ignored)
        sec_counts: Counter = Counter()
        for sh in csh:
            owner = disc.get(sh)
            if owner is not None:
                sec_counts[owner] += 1
        overlaps = sorted(
            ((n, slug) for slug, n in sec_counts.items() if n >= BLEED_MIN_SHINGLES),
            reverse=True,
        )
        if len(overlaps) >= 2:
            (ov1, s1), (ov2, s2) = overlaps[0], overlaps[1]
            severity = "high" if ov2 / ov1 >= BLEED_HIGH_RATIO else "low"
            if severity == "high":
                high += 1
            bleeds.append(
                {
                    "candidate_id": c.id,
                    "severity": severity,
                    "sections": [s1, s2],
                    "discriminative_shingles": [ov1, ov2],
                    "n_sections_touched": len(overlaps),
                }
            )
    # boundary duplication: shared sentence across adjacent candidate chunks
    boundary_dupes = 0
    boundary_examples = []
    for i in range(1, len(cand_chunks)):
        prev = {s for s in (normalize_sentence(x) for x in sentences(cand_chunks[i - 1].text)) if len(s) >= MIN_SENT_WORDS}
        cur = {s for s in (normalize_sentence(x) for x in sentences(cand_chunks[i].text)) if len(s) >= MIN_SENT_WORDS}
        shared = prev & cur
        if shared:
            boundary_dupes += len(shared)
            if len(boundary_examples) < 8:
                boundary_examples.append(
                    {
                        "a": cand_chunks[i - 1].id,
                        "b": cand_chunks[i].id,
                        "sentence": " ".join(next(iter(shared)))[:120],
                    }
                )
    return {
        "bleed_candidates": len(bleeds),
        "high_severity_bleeds": high,
        "bleeds": bleeds[:20],
        "boundary_duplicate_sentences": boundary_dupes,
        "boundary_examples": boundary_examples,
    }


def _heading_tokens(s: str) -> set[str]:
    return {t for t in tokenize(s) if t not in _STOPWORDS and len(t) > 1}


def metric_structure(gch: GoldChapter, headings: list[tuple[int, str]]) -> dict:
    level_hist: Counter = Counter(level for level, _ in headings)
    cand_head_tokens = [(_heading_tokens(h), h) for _, h in headings]
    gold_sections = gch.section_titles  # slug -> title
    missing = []
    matched_slugs = set()
    for slug, title in sorted(gold_sections.items()):
        gt = _heading_tokens(title)
        if not gt:
            matched_slugs.add(slug)
            continue
        found = False
        for ct, _h in cand_head_tokens:
            if not ct:
                continue
            inter = len(gt & ct)
            if inter and (gt <= ct or inter / len(gt) >= 0.6):
                found = True
                break
        if found:
            matched_slugs.add(slug)
        else:
            missing.append({"section": slug, "title": title})
    # extra candidate headings that match no gold section
    extra = []
    for ct, h in cand_head_tokens:
        if not ct:
            continue
        hit = False
        for slug, title in gold_sections.items():
            gt = _heading_tokens(title)
            if gt and (gt <= ct or len(gt & ct) / len(gt) >= 0.6):
                hit = True
                break
        if not hit:
            extra.append(h)
    return {
        "candidate_heading_count": len(headings),
        "level_histogram": {f"h{k}": v for k, v in sorted(level_hist.items())},
        "gold_section_count": len(gold_sections),
        "matched_sections": len(matched_slugs),
        "missing_sections": missing,
        "extra_headings": extra[:25],
    }


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------
def compute_verdict(mode: str, ch_report: dict) -> dict:
    triggers: list[str] = []
    verdict = "PASS"

    if ch_report.get("candidate_missing"):
        return {"verdict": "FAIL", "triggers": ["chapter_absent_from_candidate"]}

    recall = ch_report["recall"]["recall_mean"]
    junk = ch_report["junk"]["junk_rate"]
    mis = ch_report["mishandled"]
    rep = ch_report["repetition"]

    def fail(t):
        nonlocal verdict
        verdict = "FAIL"
        triggers.append(t)

    def warn(t):
        nonlocal verdict
        if verdict != "FAIL":
            verdict = "WARN"
        triggers.append(t)

    if recall < FAIL_RECALL:
        fail(f"recall_mean<{FAIL_RECALL}({recall})")
    elif recall < WARN_RECALL:
        warn(f"recall_mean<{WARN_RECALL}({recall})")

    if junk > FAIL_JUNK:
        fail(f"junk_rate>{FAIL_JUNK}({junk})")
    elif junk > WARN_JUNK:
        warn(f"junk_rate>{WARN_JUNK}({junk})")

    if mis["mojibake_per_1k_chars"] > FAIL_MOJIBAKE:
        fail(f"mojibake_per_1k>{FAIL_MOJIBAKE}({mis['mojibake_per_1k_chars']})")
    if mis["garbage_word_rate"] > WARN_GARBAGE:
        warn(f"garbage_word_rate>{WARN_GARBAGE}({mis['garbage_word_rate']})")
    if mis["broken_hyphen_per_1k_words"] > WARN_HYPHEN:
        warn(f"broken_hyphen>{WARN_HYPHEN}({mis['broken_hyphen_per_1k_words']})")

    if rep["repeated_12gram_excess"] > WARN_REPEAT_EXCESS:
        warn(f"repeated_12gram_excess>{WARN_REPEAT_EXCESS}({rep['repeated_12gram_excess']})")
    if rep["near_duplicate_chunk_pairs"] > 0:
        warn(f"near_dup_chunks({rep['near_duplicate_chunk_pairs']})")

    if mode == "chunks" and "bleed" in ch_report:
        bl = ch_report["bleed"]
        if bl["high_severity_bleeds"] > 0:
            fail(f"high_severity_bleeds({bl['high_severity_bleeds']})")
        elif bl["bleed_candidates"] > 0:
            warn(f"bleed_candidates({bl['bleed_candidates']})")

    if mode == "html" and "structure" in ch_report:
        st = ch_report["structure"]
        n_missing = len(st["missing_sections"])
        if n_missing > WARN_MISSING_SECT:
            warn(f"missing_sections({n_missing})")

    return {"verdict": verdict, "triggers": triggers}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def evaluate(
    gold: dict[int, GoldChapter],
    cand: dict[int, CandChapter],
    mode: str,
    chapters: list[int] | None,
) -> dict:
    if chapters is not None:
        target = sorted(set(chapters))
    else:
        target = sorted(ch for ch in gold if ch is not None)

    per_chapter: dict[str, dict] = {}
    verdicts: dict[str, dict] = {}
    for ch in target:
        gch = gold.get(ch)
        if gch is None:
            per_chapter[str(ch)] = {"error": "chapter_not_in_gold"}
            verdicts[str(ch)] = {"verdict": "FAIL", "triggers": ["chapter_not_in_gold"]}
            continue
        cch = cand.get(ch)
        if cch is None or not cch.text.strip():
            rep = {"candidate_missing": True, "gold_chunks": len(gch.chunks)}
            per_chapter[str(ch)] = rep
            verdicts[str(ch)] = compute_verdict(mode, rep)
            continue

        gold_text = " ".join(c.text for c in gch.chunks)
        ch_rep: dict = {
            "candidate_missing": False,
            "recall": metric_recall(gch, cch.text),
            "junk": metric_junk(gch, cch.text),
            "mishandled": metric_mishandled(cch.text),
            "repetition": metric_repetition(gch, cch.text, cch.chunks),
            # How much math-representation folding was applied per side, so an
            # audit can attribute junk/garbage reduction to notation, not defect.
            "normalization": {
                "candidate": count_math_folds(cch.text),
                "gold": count_math_folds(gold_text),
            },
        }
        if mode == "chunks":
            ch_rep["bleed"] = metric_bleed(gch, cch.chunks)
            ch_rep["candidate_chunk_count"] = len(cch.chunks)
            ch_rep["fallback_assigned_chunks"] = cch.fallback_chunks
        if mode == "html":
            ch_rep["structure"] = metric_structure(gch, cch.headings)
        v = compute_verdict(mode, ch_rep)
        ch_rep["verdict"] = v["verdict"]
        ch_rep["triggers"] = v["triggers"]
        per_chapter[str(ch)] = ch_rep
        verdicts[str(ch)] = v

    # overall roll-up
    recalls = [
        r["recall"]["recall_mean"]
        for r in per_chapter.values()
        if not r.get("candidate_missing") and "recall" in r
    ]
    junks = [
        r["junk"]["junk_rate"]
        for r in per_chapter.values()
        if not r.get("candidate_missing") and "junk" in r
    ]
    vcounts = Counter(v["verdict"] for v in verdicts.values())
    # roll up the per-side folding counts across evaluated chapters
    norm_totals = {
        side: {
            "latex_commands": 0,
            "unicode_symbols": 0,
            "digit_letter_fusions": 0,
            "part_markers": 0,
        }
        for side in ("candidate", "gold")
    }
    for r in per_chapter.values():
        norm = r.get("normalization")
        if not norm:
            continue
        for side in ("candidate", "gold"):
            for k, v in norm.get(side, {}).items():
                norm_totals[side][k] += v
    overall = {
        "chapters_evaluated": len(target),
        "mean_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "worst_recall": round(min(recalls), 4) if recalls else 0.0,
        "mean_junk_rate": round(sum(junks) / len(junks), 4) if junks else 0.0,
        "verdict_counts": dict(vcounts),
        "normalization": norm_totals,
        "overall_verdict": "FAIL"
        if vcounts.get("FAIL")
        else ("WARN" if vcounts.get("WARN") else "PASS"),
    }
    return {"per_chapter": per_chapter, "overall": overall, "verdicts": verdicts}


def print_summary(report: dict, mode: str) -> None:
    ov = report["overall"]
    print("=" * 72)
    print(f"GOLD-COMPARE  mode={mode}  overall={ov['overall_verdict']}")
    print(
        f"  chapters={ov['chapters_evaluated']}  mean_recall={ov['mean_recall']}"
        f"  worst_recall={ov['worst_recall']}  mean_junk={ov['mean_junk_rate']}"
        f"  verdicts={ov['verdict_counts']}"
    )
    print("-" * 72)
    for ch in sorted(report["per_chapter"], key=lambda x: int(x) if x.lstrip('-').isdigit() else 0):
        r = report["per_chapter"][ch]
        v = report["verdicts"][ch]
        if r.get("candidate_missing"):
            print(f"  ch{ch:>3}  {v['verdict']:<4}  candidate missing "
                  f"({r.get('gold_chunks','?')} gold chunks)")
            continue
        if "recall" not in r:
            print(f"  ch{ch:>3}  {v['verdict']:<4}  {r.get('error','')}")
            continue
        line = (
            f"  ch{ch:>3}  {v['verdict']:<4}  recall={r['recall']['recall_mean']:.3f}"
            f"  junk={r['junk']['junk_rate']:.3f}"
            f"  moji/1k={r['mishandled']['mojibake_per_1k_chars']:.2f}"
        )
        if mode == "chunks":
            line += f"  bleeds={r['bleed']['bleed_candidates']}({r['bleed']['high_severity_bleeds']}hi)"
        if mode == "html":
            line += f"  miss_sect={len(r['structure']['missing_sections'])}"
        print(line)
        if v["triggers"]:
            print(f"          -> {', '.join(v['triggers'])}")
    print("=" * 72)


def parse_map(entries: list[str] | None) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for e in entries or []:
        if "=" not in e:
            raise SystemExit(f"--map expects chNN=path, got: {e}")
        key, path = e.split("=", 1)
        ch = chapter_from_text(key)
        if ch is None:
            raise SystemExit(f"--map key must contain chNN, got: {key}")
        out[ch] = Path(path)
    return out


def parse_chapters(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gold-comparison QA harness for SemantiK conversions")
    ap.add_argument("--gold", required=True, type=Path, help="gold chunk_v4 chunks.jsonl")
    ap.add_argument("--candidate-html", type=Path, help="dir of candidate accessible-HTML files")
    ap.add_argument("--candidate-chunks", type=Path, help="candidate chunk_v4 chunks.jsonl")
    ap.add_argument("--map", action="append", help="explicit chNN=path.html mapping (repeatable)")
    ap.add_argument("--chapters", help="comma/range list, e.g. 1,2,5-7 (default: all gold chapters)")
    ap.add_argument("--json-out", type=Path, help="write full JSON report here")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not args.candidate_html and not args.candidate_chunks:
        ap.error("provide at least one of --candidate-html / --candidate-chunks")

    gold = load_gold(args.gold)
    chapters = parse_chapters(args.chapters)
    explicit_map = parse_map(args.map)

    results = {}
    if args.candidate_chunks:
        cand = load_candidate_chunks(args.candidate_chunks, gold)
        results["chunks"] = evaluate(gold, cand, "chunks", chapters)
    if args.candidate_html or explicit_map:
        cand = load_candidate_html(args.candidate_html, explicit_map)
        results["html"] = evaluate(gold, cand, "html", chapters)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.time(),
        "gold_path": str(args.gold),
        "candidate": {
            "html_dir": str(args.candidate_html) if args.candidate_html else None,
            "chunks": str(args.candidate_chunks) if args.candidate_chunks else None,
            "explicit_map": {f"ch{k:02d}": str(v) for k, v in explicit_map.items()},
        },
        "modes": {},
    }
    # flatten single-mode into top-level per_chapter/overall/verdicts for convenience
    for mode, res in results.items():
        report["modes"][mode] = res
        print_summary(res, mode)
    # convenience top-level = last/primary mode
    primary = "chunks" if "chunks" in results else "html"
    if primary in results:
        report["per_chapter"] = results[primary]["per_chapter"]
        report["overall"] = results[primary]["overall"]
        report["verdicts"] = results[primary]["verdicts"]
        report["primary_mode"] = primary

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report -> {args.json_out}")

    overall_v = report.get("overall", {}).get("overall_verdict", "PASS")
    return 0 if overall_v != "FAIL" else 2


if __name__ == "__main__":
    sys.exit(main())

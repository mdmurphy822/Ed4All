#!/usr/bin/env python3
"""Deterministic, domain-agnostic structure / accessibility scorecard for
SemantiK-converted (accessible) HTML.

SemantiK converts *any* human-readable source (papers, reports, novels,
manuals, textbooks) to accessible HTML. ``scripts/harness/gold_compare.py`` measures
*text fidelity* against a gold chunkset; this harness measures the orthogonal
axis: **structural and presentation quality of the emitted HTML itself**, with
no reference document required. It is deterministic and CPU-only -- stdlib +
BeautifulSoup/lxml (already repo deps); no LLM, no GPU, no embeddings.

The scorecard is calibrated on the SemantiK flagship conversion examples (diverse
gold-quality source->accessible.html pairs). It is *domain-agnostic*: it never
looks at subject-matter vocabulary, only at the self-referential contract of an
accessible document -- "does the markup deliver the structure it declares, and
is it clean, navigable, and captioned?".

Dimensions (see DIMENSIONS below for the config):

  1. DELIVER-WHAT-YOU-DECLARE (scored, core / self-referential)
       Two arms, chosen per-document by whether the doc carries structural
       role annotations.
       (a) Annotation arm: any element carrying ``data-semantik-block-role``
           (legacy corpora: ``data-dart-block-role``) or an ARIA ``role``
           of ``list`` / ``table`` / ``definition_list`` /
           ``figure`` MUST contain the matching semantic element
           (``<ul>``/``<ol>`` / ``<table>`` / ``<dl>`` / ``<figure>``|``<img alt>``).
           Metric: delivery rate per declared role type.
       (b) Generic arm (docs with no such annotations -- e.g. the exemplars):
           conservatively detect *high-confidence* content shapes left in the
           visible prose that a real converter would have promoted to semantic
           markup -- a run of >=N consecutive pipe-delimited rows (an un-promoted
           table) or a run of >=N consecutive list-marker lines (an un-promoted
           list). Score is the fraction of such shapes that are NOT left orphaned
           in prose. No shapes + no annotations -> dimension is N/A (score 1.0).
  2. HEADING HIERARCHY (scored): exactly one h1; no level skips; non-empty
       headings; no raw LaTeX/markdown markup in heading text.
  3. LANDMARKS & NAV (scored): single ``<main>``; skip link present and its
       target resolves; ``<nav>`` elements labelled; no duplicate ``id``s.
       TOC-with-anchors presence is reported but NOT scored (fragments lack it).
  4. TEXT CLEANLINESS (scored, regression guard): visible-text leakage of LaTeX
       commands outside math delimiters, markdown fences/pipe-tables, literal
       HTML entity artifacts, and U+FFFD replacement chars. Rate per 1k words.
  5. IMAGE / FIGURE A11Y (scored): every ``<img>`` has non-trivial alt (>=15
       chars) or is explicitly decorative; figure/figcaption pairing; alt-text
       descriptiveness proxy (not just a filename or the word "image").
  6. MATH ACCESSIBILITY (report-only): count of raw ``$``-delimited LaTeX runs
       (signals a MathJax/MathML assistive need downstream) + MathML presence.
  7. LANGUAGE / META (scored: lang + title; report-only: description/author).

Usage::

    python3 scripts/harness/structure_scorecard.py --html path/to/accessible.html
    python3 scripts/harness/structure_scorecard.py --html some/dir/ --json-out report.json
    python3 scripts/harness/structure_scorecard.py --html gold_dir/ --baseline

``--html`` accepts a single ``.html`` file or a directory (all ``*.html`` under
it, recursively). ``--json-out`` writes the full machine-readable report.
``--baseline`` is a labelling hint for the human-readable header only (marks the
run as a baseline/gap measurement rather than a gate check).

This module is intentionally free of any hardcoded corpus/course/input path --
every path is argument-driven -- so it is safe to track in git.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup
except Exception as exc:  # pragma: no cover - dependency guard
    sys.stderr.write(
        "structure_scorecard requires beautifulsoup4 (and lxml). "
        f"Import failed: {exc}\n"
    )
    raise

# ---------------------------------------------------------------------------
# Configuration -- single source of truth for weights and thresholds.
# Tuned so every SemantiK exemplar PASSes every *scored* dimension (see the module
# docstring / the plans doc for the calibration table). Change here, re-run the
# calibration in scripts/tests to confirm the exemplars stay green.
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    # Composite = weighted mean of the scored dimensions' [0,1] scores.
    "weights": {
        "deliver": 0.30,
        "heading": 0.20,
        "landmarks": 0.20,
        "cleanliness": 0.15,
        "image_a11y": 0.15,
        # language is folded in at a small fixed weight below; math + meta are
        # report-only and never enter the composite.
        "language": 0.05,
    },
    # Verdict bands (apply to the composite AND to each scored dimension).
    "verdict": {
        "pass": 0.90,   # >= pass  -> PASS
        "warn": 0.70,   # >= warn  -> WARN, else FAIL
    },
    # Dimension 1 generic arm: only flag *runs* of a shape (conservative).
    "generic_shape_min_run": 3,
    # Dimension 4 cleanliness: leakage artifacts per 1000 visible words.
    # score 1.0 at 0/1k, linearly -> 0.0 at fail rate.
    "cleanliness_warn_per_1k": 0.5,
    "cleanliness_fail_per_1k": 5.0,
    # Dimension 5 image a11y.
    "alt_min_len": 15,
}

# Structural role names we hold to the "declare -> deliver" contract, mapped to
# the semantic element(s) that satisfy them.
_ROLE_MATCHERS: dict[str, tuple[str, ...]] = {
    "list": ("ul", "ol"),
    "table": ("table",),
    "definition_list": ("dl",),
    "figure": ("figure", "img"),
}

# LaTeX command leakage in *visible* text (a superset of the classes we have
# historically fixed). Deliberately conservative -- only backslash-command
# tokens and \[ \] display delimiters, so ordinary prose backslashes (rare) and
# inline $..$ math (dimension 6, report-only) do not trip this.
_LATEX_LEAK = re.compile(
    r"\\(?:textbf|textit|texttt|emph|begin|end|hline|cline|multicolumn|frac|"
    r"sqrt|section|subsection|subsubsection|paragraph|item|cite|ref|eqref|"
    r"label|mathbf|mathrm|mathcal|left|right|cdot|times|alpha|beta|gamma|"
    r"delta|theta|lambda|sigma|sum|int|infty|partial|nabla|vec|hat|bar|"
    r"overline|underline|quad|qquad|newline|noindent)\b"
    r"|\\\[|\\\]"
)
# Markdown fences / heading hashes / pipe-table rows leaking into visible text.
_MD_FENCE = re.compile(r"```|~~~")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")
# A markdown table row leaked into prose: canonically starts with a leading
# pipe and has >=1 further pipe (cell delimiter). This deliberately EXCLUDES
# prose that merely contains pipes mid-sentence -- bra-ket notation (|00>),
# conditional probability P(x|y), OCR caption fragments -- which are legitimate
# content, not un-promoted tables.
_PIPE_TABLE_ROW = re.compile(r"^\s{0,6}\|.*\|")
_LITERAL_ENTITY = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#\d+|#x[0-9A-Fa-f]+);")
_FFFD = "�"
# A list-marker line: bullet or enumerated.
_LIST_MARK = re.compile(r"^\s{0,6}(?:[-*•‣◦⁃]\s+|\(?\d{1,3}[.)]\s+|[a-z]\)\s+)")
# Raw markup inside heading text.
_HEADING_MARKUP = re.compile(r"\\[a-zA-Z]+\{|```|~~~|^\s*#{1,6}\s|\|.*\|")
# $-delimited inline/display math run (report-only dimension 6). Bounded (<=300
# chars, single line) so the report-only raw-math COUNT is not distorted by a
# runaway match across unbalanced ``$``.
_MATH_DELIM = re.compile(r"(?<!\\)\$(?!\$)[^$\n]{1,300}?(?<!\\)\$|\\\((?:.|\n)+?\\\)|\\\[(?:.|\n)+?\\\]")
# Cleanliness (dimension 4) math strip — removes *delimited* math of ANY length
# before counting visible LaTeX leakage, so properly-delimited math (which
# MathJax renders and is therefore NOT visible garbage) never counts as a leak.
# Uncapped + DOTALL: a long/malformed but still ``$…$``-delimited answer-key span
# (a real conversion artifact) is math, not leakage. Applied as SEQUENTIAL passes
# (display ``$$…$$`` FULLY before single ``$…$``) so a single ``$`` adjacent to a
# ``$$`` pair cannot consume one of its dollars and desync the pairing across the
# whole document. ``(?<!\\)`` keeps an escaped ``\$`` (currency) from opening a
# span — so the round-7b adapter emit of preserved currency as ``\$5`` never
# opens a phantom inline span here (and, unpaired, never desyncs the pairing:
# an escaped ``\$`` opens nothing, and the following genuine ``$…$`` pairs cleanly).
_CLEAN_MATH_STRIP_PASSES = (
    re.compile(r"(?<!\\)\$\$.*?(?<!\\)\$\$", re.DOTALL),
    # Forbid a nested same-opener so an orphan ``\[`` isn't swallowed as content.
    re.compile(r"\\\[(?:(?!\\\[).)*?\\\]", re.DOTALL),
    re.compile(r"\\\((?:(?!\\\().)*?\\\)", re.DOTALL),
    re.compile(r"(?<!\\)\$[^$]*\$", re.DOTALL),
)


def _strip_delimited_math(text: str) -> str:
    """Strip delimited math in sequential passes (display before inline)."""
    for pat in _CLEAN_MATH_STRIP_PASSES:
        text = pat.sub(" ", text)
    return text
# Trivial alt text (filename-only or a generic placeholder word).
_TRIVIAL_ALT = re.compile(
    r"^(?:image|images|figure|fig|picture|photo|graphic|img|icon|logo|untitled|"
    r"screenshot|diagram)?[\s._-]*$",
    re.IGNORECASE,
)
_FILENAMEISH = re.compile(r"^[\w./\\-]+\.(?:png|jpe?g|gif|svg|webp|bmp|tiff?)$", re.IGNORECASE)

_HEADING_RE = re.compile(r"^h[1-6]$")


def _clamp(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _verdict(score: float, cfg: dict[str, Any] | None = None) -> str:
    v = (cfg or CONFIG)["verdict"]
    if score >= v["pass"]:
        return "PASS"
    if score >= v["warn"]:
        return "WARN"
    return "FAIL"


@dataclass
class DimensionResult:
    name: str
    scored: bool
    score: float  # [0,1]; for report-only dims this is informational only
    verdict: str
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class FileResult:
    path: str
    composite: float
    verdict: str
    dimensions: dict[str, DimensionResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "composite": round(self.composite, 4),
            "verdict": self.verdict,
            "dimensions": {
                k: {
                    "scored": d.scored,
                    "score": round(d.score, 4),
                    "verdict": d.verdict,
                    "details": d.details,
                    "notes": d.notes,
                }
                for k, d in self.dimensions.items()
            },
        }


# ---------------------------------------------------------------------------
# Visible-text extraction that excludes the elements dimension 1's generic arm
# is allowed to "already count" as delivered structure (tables/lists/pre/code).
# ---------------------------------------------------------------------------
def _visible_lines_excluding_structure(body) -> list[str]:
    clone = BeautifulSoup(str(body), "lxml")
    for el in clone.find_all(["table", "ul", "ol", "pre", "code", "figure", "script", "style"]):
        el.decompose()
    text = clone.get_text("\n")
    return [ln.rstrip() for ln in text.split("\n")]


def _visible_text(body) -> str:
    clone = BeautifulSoup(str(body), "lxml")
    for el in clone.find_all(["script", "style"]):
        el.decompose()
    return clone.get_text(" ")


def _visible_text_newlines(body) -> str:
    clone = BeautifulSoup(str(body), "lxml")
    for el in clone.find_all(["script", "style"]):
        el.decompose()
    return clone.get_text("\n")


def _max_consecutive_run(lines: list[str], pred) -> tuple[int, int]:
    """Return (max_run_length, number_of_runs>=min_run) is computed by caller;
    here return (max_run, count_of_matching_lines)."""
    run = 0
    best = 0
    for ln in lines:
        if ln.strip() and pred(ln):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best, sum(1 for ln in lines if ln.strip() and pred(ln))


def _pipe_run_line_count(lines: list[str]) -> int:
    """Total lines that belong to a run of >=2 consecutive pipe-table rows.

    A pipe-table row canonically starts with a leading ``|`` (see
    ``_PIPE_TABLE_ROW``). Isolated pipe lines (bra-ket notation, conditional
    probabilities, OCR caption fragments) are NOT counted -- only genuine
    multi-row markdown-table leakage is."""
    def is_pipe(ln: str) -> bool:
        return bool(_PIPE_TABLE_ROW.match(ln))

    total = 0
    run = 0
    for ln in lines:
        if is_pipe(ln):
            run += 1
        else:
            if run >= 2:
                total += run
            run = 0
    if run >= 2:
        total += run
    return total


def _count_runs(lines: list[str], pred, min_run: int) -> int:
    """Count how many distinct runs of >= min_run consecutive matching lines."""
    run = 0
    runs = 0
    for ln in lines:
        if ln.strip() and pred(ln):
            run += 1
            if run == min_run:
                runs += 1
        else:
            run = 0
    return runs


# ---------------------------------------------------------------------------
# Dimension 1: DELIVER-WHAT-YOU-DECLARE
# ---------------------------------------------------------------------------
def _dim_deliver(soup, body, cfg) -> DimensionResult:
    notes: list[str] = []
    # Gather declared-structure blocks from the block-role annotation (live
    # SemantiK ``data-semantik-block-role``; legacy corpora ``data-dart-block-role``)
    # or an ARIA role.
    declared: dict[str, list] = {r: [] for r in _ROLE_MATCHERS}
    for el in body.find_all(True):
        role = el.get("data-semantik-block-role") or el.get("data-dart-block-role")
        if role not in _ROLE_MATCHERS:
            role = None
        if role is None:
            aria = el.get("role")
            if aria in _ROLE_MATCHERS:
                role = aria
        if role is not None:
            declared[role].append(el)

    total_declared = sum(len(v) for v in declared.values())

    # Blind-spot guard (report-only): a block whose declared list/table/
    # definition_list role was honestly reconciled to <p> by the adapter (no
    # high-confidence shape delivered) carries ``data-semantik-demoted-role``
    # (legacy corpora: ``data-dart-demoted-role``). That reconciliation drops the
    # role from the block-role annotation, so it is INVISIBLE to the deliver
    # metric above (deliver stays high even when a scrub eats every pipe row and
    # mass-demotes real tables). Surfacing the demotion count here catches that
    # regression class. Never enters the score.
    demoted_by_role: dict[str, int] = {}
    for el in body.find_all(True):
        dr = el.get("data-semantik-demoted-role") or el.get("data-dart-demoted-role")
        if dr in _ROLE_MATCHERS:
            demoted_by_role[dr] = demoted_by_role.get(dr, 0) + 1
    demoted_declarations = sum(demoted_by_role.values())

    if total_declared > 0:
        arm = "annotation"
        per_role: dict[str, dict[str, int]] = {}
        matched_total = 0
        for role, els in declared.items():
            if not els:
                continue
            matchers = _ROLE_MATCHERS[role]
            m = 0
            for el in els:
                # For a wrapper element (e.g. <section role=table>) the matching
                # semantic element must appear as a descendant. For an element
                # that IS the semantic element, find(...) still finds it via
                # find on the element's own subtree, so also test the tag itself.
                if el.name in matchers:
                    found = True
                elif role == "figure":
                    img = el.find("img")
                    found = el.find("figure") is not None or (
                        img is not None and len((img.get("alt") or "").strip()) > 0
                    )
                else:
                    found = el.find(list(matchers)) is not None
                if found:
                    m += 1
            per_role[role] = {"declared": len(els), "delivered": m}
            matched_total += m
        score = matched_total / total_declared if total_declared else 1.0
        details = {
            "arm": arm,
            "declared_total": total_declared,
            "delivered_total": matched_total,
            "per_role": per_role,
            "demoted_declarations": demoted_declarations,
            "demoted_by_role": demoted_by_role,
        }
        if demoted_declarations:
            notes.append(
                f"{demoted_declarations} declared structural role(s) demoted to "
                f"paragraph (no shape delivered): {demoted_by_role}"
            )
        return DimensionResult(
            "deliver", True, score, _verdict(score, cfg), details, notes
        )

    # Generic arm: conservative content-shape detection.
    arm = "generic"
    lines = _visible_lines_excluding_structure(body)
    min_run = cfg["generic_shape_min_run"]

    def _is_pipe(ln: str) -> bool:
        return bool(_PIPE_TABLE_ROW.match(ln))

    orphan_table_runs = _count_runs(lines, _is_pipe, min_run)
    orphan_list_runs = _count_runs(lines, lambda ln: bool(_LIST_MARK.match(ln)), min_run)

    # Semantic structure already present in the doc (credit for delivery).
    present_tables = len(body.find_all("table"))
    present_lists = len(body.find_all(["ul", "ol"]))

    orphan_shapes = orphan_table_runs + orphan_list_runs
    if orphan_shapes == 0:
        score = 1.0
        if present_tables == 0 and present_lists == 0:
            notes.append(
                "no structural signals (no annotations, no orphan shapes, no "
                "tables/lists) -- dimension is N/A, scored 1.0"
            )
    else:
        # Denominator = orphaned shapes + already-delivered semantic groups, so
        # a doc that mostly delivered its structure is not punished for one
        # stray run. Conservative: only high-confidence runs count as orphans.
        delivered_groups = present_tables + present_lists
        score = delivered_groups / (delivered_groups + orphan_shapes)
        notes.append(
            f"generic arm flagged {orphan_table_runs} un-promoted pipe-table run(s) "
            f"and {orphan_list_runs} un-promoted list run(s)"
        )
    details = {
        "arm": arm,
        "orphan_pipe_table_runs": orphan_table_runs,
        "orphan_list_runs": orphan_list_runs,
        "present_tables": present_tables,
        "present_lists": present_lists,
        "min_run": min_run,
        "demoted_declarations": demoted_declarations,
        "demoted_by_role": demoted_by_role,
    }
    if demoted_declarations:
        notes.append(
            f"{demoted_declarations} declared structural role(s) demoted to "
            f"paragraph (no shape delivered): {demoted_by_role}"
        )
    return DimensionResult("deliver", True, score, _verdict(score, cfg), details, notes)


# ---------------------------------------------------------------------------
# Dimension 2: HEADING HIERARCHY
# ---------------------------------------------------------------------------
def _dim_heading(soup, body, cfg) -> DimensionResult:
    headings = body.find_all(_HEADING_RE)
    notes: list[str] = []
    if not headings:
        notes.append("document has no headings")
        # A doc with zero headings is structurally weak but not a hard fail for
        # a bare fragment; score conservatively at WARN.
        return DimensionResult(
            "heading", True, cfg["verdict"]["warn"], "WARN",
            {"heading_count": 0, "h1_count": 0}, notes,
        )

    levels = [int(h.name[1]) for h in headings]
    h1_count = levels.count(1)

    # Skips (h2 -> h4).
    skips = []
    prev = None
    transitions = 0
    for lv in levels:
        if prev is not None:
            transitions += 1
            if lv > prev + 1:
                skips.append((prev, lv))
        prev = lv

    empty = [h for h in headings if not h.get_text(strip=True)]
    markuped = [
        h.get_text(" ", strip=True)[:60]
        for h in headings
        if _HEADING_MARKUP.search(h.get_text(" ", strip=True))
    ]

    h1_ok = 1.0 if h1_count == 1 else 0.0
    if h1_count == 0:
        notes.append("no <h1> present")
    elif h1_count > 1:
        notes.append(f"{h1_count} <h1> elements (expected exactly 1)")
    no_skip_ratio = 1.0 - (len(skips) / transitions) if transitions else 1.0
    if skips:
        notes.append(f"heading level skip(s): {skips[:5]}")
    nonempty_ratio = 1.0 - (len(empty) / len(headings))
    if empty:
        notes.append(f"{len(empty)} empty heading(s)")
    clean_ratio = 1.0 - (len(markuped) / len(headings))
    if markuped:
        notes.append(f"{len(markuped)} heading(s) with raw markup, e.g. {markuped[:3]}")

    score = (h1_ok + no_skip_ratio + nonempty_ratio + clean_ratio) / 4.0
    details = {
        "heading_count": len(headings),
        "h1_count": h1_count,
        "level_skips": skips,
        "empty_headings": len(empty),
        "markup_in_headings": len(markuped),
        "level_histogram": {f"h{n}": levels.count(n) for n in range(1, 7) if levels.count(n)},
    }
    return DimensionResult("heading", True, score, _verdict(score, cfg), details, notes)


# ---------------------------------------------------------------------------
# Dimension 3: LANDMARKS & NAV
# ---------------------------------------------------------------------------
def _dim_landmarks(soup, body, cfg) -> DimensionResult:
    notes: list[str] = []
    mains = body.find_all("main") + body.find_all(attrs={"role": "main"})
    # Dedupe (a <main role=main> would double count).
    mains = list({id(m): m for m in mains}.values())
    main_ok = 1.0 if len(mains) == 1 else 0.0
    if len(mains) == 0:
        notes.append("no <main> landmark")
    elif len(mains) > 1:
        notes.append(f"{len(mains)} main landmarks (expected exactly 1)")

    # Skip link: an <a href="#..."> whose text mentions skip, target resolves.
    skip_href = None
    for a in body.find_all("a", href=True):
        if a["href"].startswith("#") and "skip" in a.get_text(strip=True).lower():
            skip_href = a["href"]
            break
    skip_ok = 0.0
    target_resolves = None
    if skip_href:
        tid = skip_href[1:]
        target = soup.find(id=tid) or soup.find(attrs={"name": tid})
        target_resolves = target is not None
        skip_ok = 1.0 if target_resolves else 0.0
        if not target_resolves:
            notes.append(f"skip link target #{tid} does not resolve")
    else:
        notes.append("no skip-to-content link")

    # Nav labelling.
    navs = body.find_all("nav")
    labelled = [n for n in navs if n.get("aria-label") or n.get("aria-labelledby")]
    nav_label_ratio = (len(labelled) / len(navs)) if navs else 1.0
    if navs and len(labelled) < len(navs):
        notes.append(f"{len(navs) - len(labelled)} of {len(navs)} <nav> unlabelled")

    # Duplicate ids (over the whole document).
    ids = [t.get("id") for t in soup.find_all(id=True)]
    seen: dict[str, int] = {}
    for i in ids:
        seen[i] = seen.get(i, 0) + 1
    dup_ids = sorted(i for i, c in seen.items() if c > 1)
    dupid_ok = 1.0 if not dup_ids else 0.0
    if dup_ids:
        notes.append(f"{len(dup_ids)} duplicate id(s): {dup_ids[:5]}")

    # TOC presence -- report only, NOT scored.
    toc_present = bool(
        [n for n in navs if n.find("a", href=lambda h: h and h.startswith("#"))]
    )

    score = (main_ok + skip_ok + nav_label_ratio + dupid_ok) / 4.0
    details = {
        "main_count": len(mains),
        "skip_link": skip_href,
        "skip_target_resolves": target_resolves,
        "nav_count": len(navs),
        "nav_labelled": len(labelled),
        "duplicate_ids": dup_ids,
        "toc_with_anchors_present": toc_present,  # report-only
    }
    return DimensionResult("landmarks", True, score, _verdict(score, cfg), details, notes)


# ---------------------------------------------------------------------------
# Dimension 4: TEXT CLEANLINESS
# ---------------------------------------------------------------------------
def _dim_cleanliness(soup, body, cfg) -> DimensionResult:
    notes: list[str] = []
    text = _visible_text(body)
    # Newline-joined variant so per-line detectors (pipe-table runs, markdown
    # heading hashes) see genuine line boundaries between block elements.
    text_lines = _visible_text_newlines(body)
    words = max(1, len(text.split()))

    # Strip delimited-math spans (of ANY length) before counting LaTeX leakage,
    # so legitimate math (dimension 6, report-only) is not double-penalised here
    # and a long-but-delimited answer-key span is not mis-counted as leakage.
    text_no_math = _strip_delimited_math(text)

    latex = len(_LATEX_LEAK.findall(text_no_math))
    fences = len(_MD_FENCE.findall(text))
    md_head = len(_MD_HEADING.findall(text_lines))
    # A genuine leaked markdown table is ALWAYS a multi-row run; single lines
    # carrying pipes are legitimate content (bra-ket notation |0>, conditional
    # probability P(x|y), OCR captions), so only count pipe lines inside a run
    # of >=2 consecutive pipe rows.
    pipe_lines = _pipe_run_line_count(text_lines.split("\n"))
    fffd = text.count(_FFFD)
    # literal entities only count when the *decoded* text still shows "&nbsp;"
    # style artifacts (i.e. double-encoded in the source).
    literal_entities = len(_LITERAL_ENTITY.findall(text))

    artifacts = latex + fences + md_head + pipe_lines + fffd + literal_entities
    per_1k = artifacts / words * 1000.0

    warn_rate = cfg["cleanliness_warn_per_1k"]
    fail_rate = cfg["cleanliness_fail_per_1k"]
    if per_1k <= warn_rate:
        score = 1.0 - (per_1k / warn_rate) * 0.10  # tiny slope in the clean band
    else:
        # linear from (warn_rate -> 0.90) down to (fail_rate -> 0.0)
        span = max(1e-9, fail_rate - warn_rate)
        score = 0.90 * (1.0 - (per_1k - warn_rate) / span)
    score = _clamp(score)

    if artifacts:
        notes.append(
            f"{artifacts} leakage artifact(s) over {words} words "
            f"({per_1k:.2f}/1k): latex={latex} fences={fences} md_head={md_head} "
            f"pipe_lines={pipe_lines} fffd={fffd} entities={literal_entities}"
        )
    details = {
        "words": words,
        "latex_commands": latex,
        "markdown_fences": fences,
        "markdown_headings": md_head,
        "pipe_table_lines": pipe_lines,
        "replacement_chars": fffd,
        "literal_entities": literal_entities,
        "artifacts_per_1k_words": round(per_1k, 4),
    }
    return DimensionResult("cleanliness", True, score, _verdict(score, cfg), details, notes)


# ---------------------------------------------------------------------------
# Dimension 5: IMAGE / FIGURE A11Y
# ---------------------------------------------------------------------------
def _dim_image_a11y(soup, body, cfg) -> DimensionResult:
    notes: list[str] = []
    imgs = body.find_all("img")
    if not imgs:
        notes.append("no <img> elements -- dimension N/A, scored 1.0")
        return DimensionResult(
            "image_a11y", True, 1.0, "PASS",
            {"img_count": 0, "figure_count": len(body.find_all("figure"))}, notes,
        )

    min_len = cfg["alt_min_len"]
    good = 0
    decorative = 0
    trivial = 0
    missing = 0
    for im in imgs:
        alt = (im.get("alt") or "").strip()
        is_decorative = (
            im.get("role") == "presentation"
            or im.get("aria-hidden") == "true"
            or (im.has_attr("alt") and alt == "" and im.get("role") == "presentation")
        )
        if is_decorative:
            decorative += 1
            good += 1  # explicitly-decorative is a correct a11y choice
            continue
        if not im.has_attr("alt") or alt == "":
            missing += 1
            continue
        if _TRIVIAL_ALT.match(alt) or _FILENAMEISH.match(alt):
            trivial += 1
            continue
        if len(alt) >= min_len:
            good += 1
        else:
            trivial += 1  # short, non-empty, non-decorative -> weak alt

    alt_ok_ratio = good / len(imgs)
    if missing:
        notes.append(f"{missing} <img> missing/empty alt (and not decorative)")
    if trivial:
        notes.append(f"{trivial} <img> with trivial/short alt (< {min_len} chars or filename-ish)")

    # Figure/figcaption pairing (report-weighted, small).
    figures = body.find_all("figure")
    with_caption = [f for f in figures if f.find("figcaption")]
    fig_pair_ratio = (len(with_caption) / len(figures)) if figures else 1.0
    if figures and len(with_caption) < len(figures):
        notes.append(f"{len(figures) - len(with_caption)} <figure> without <figcaption>")

    score = 0.85 * alt_ok_ratio + 0.15 * fig_pair_ratio
    details = {
        "img_count": len(imgs),
        "good_alt": good,
        "decorative": decorative,
        "trivial_alt": trivial,
        "missing_alt": missing,
        "figure_count": len(figures),
        "figures_with_caption": len(with_caption),
        "alt_ok_ratio": round(alt_ok_ratio, 4),
        "figure_pair_ratio": round(fig_pair_ratio, 4),
    }
    return DimensionResult("image_a11y", True, score, _verdict(score, cfg), details, notes)


# ---------------------------------------------------------------------------
# Dimension 6: MATH ACCESSIBILITY (report-only)
# ---------------------------------------------------------------------------
def _dim_math(soup, body, cfg) -> DimensionResult:
    text = _visible_text(body)
    math_runs = len(_MATH_DELIM.findall(text))
    mathml = len(body.find_all("math"))
    notes = []
    if math_runs and not mathml:
        notes.append(
            f"{math_runs} raw $-delimited LaTeX math run(s) and no MathML -- "
            "downstream MathJax/assistive rendering recommended"
        )
    details = {"raw_math_runs": math_runs, "mathml_elements": mathml}
    # Report-only: score reflects the observation but never enters composite.
    score = 1.0 if (mathml or math_runs == 0) else 0.5
    return DimensionResult("math", False, score, "REPORT", details, notes)


# ---------------------------------------------------------------------------
# Dimension 7: LANGUAGE / META (lang + title scored; desc/author report-only)
# ---------------------------------------------------------------------------
def _dim_language(soup, body, cfg) -> DimensionResult:
    notes: list[str] = []
    html_el = soup.find("html")
    lang = html_el.get("lang") if html_el else None
    lang_ok = 1.0 if (lang and lang.strip()) else 0.0
    if not lang_ok:
        notes.append("<html> missing a non-empty lang attribute")

    title_el = soup.find("title")
    title = title_el.get_text(strip=True) if title_el else ""
    title_ok = 1.0 if title else 0.0
    if not title_ok:
        notes.append("missing or empty <title>")

    # Report-only meta.
    def _meta(name):
        m = soup.find("meta", attrs={"name": name})
        return (m.get("content") if m else None) or None

    description = _meta("description")
    author = _meta("author")

    score = (lang_ok + title_ok) / 2.0
    details = {
        "lang": lang,
        "title_present": bool(title),
        "title_length": len(title),
        "meta_description_present": bool(description),  # report-only
        "meta_author_present": bool(author),            # report-only
    }
    return DimensionResult("language", True, score, _verdict(score, cfg), details, notes)


# ---------------------------------------------------------------------------
# Dimension 8: PEDAGOGY (report-only): tally of ``data-semantik-unit`` annotations
# ---------------------------------------------------------------------------
def _dim_pedagogy(soup, body, cfg) -> DimensionResult:
    """Count ``data-semantik-unit`` pedagogical-unit annotations per document.

    Report-only: SemantiK-authored course pages carry ``data-semantik-unit``
    (legacy corpora: ``data-dart-unit``) markers (e.g. ``worked_example`` /
    ``exercise_set`` / ``section_opener``); exemplars, fiction, and any
    non-SemantiK doc carry none. This facet simply tallies them so the
    pedagogical shape is visible in the report. It is never scored and never
    enters the composite (a doc with zero units is a no-op, NEVER a penalty)."""
    by_type: dict[str, int] = {}
    for el in body.find_all(True):
        unit = (
            el.get("data-semantik-unit") or el.get("data-dart-unit") or ""
        ).strip()
        if not unit:
            continue
        by_type[unit] = by_type.get(unit, 0) + 1
    units_by_type = {k: by_type[k] for k in sorted(by_type)}
    total = sum(units_by_type.values())

    notes: list[str] = []
    if total == 0:
        notes.append(
            "no data-semantik-unit annotations (exemplar/fiction/non-SemantiK) "
            "-- pedagogy facet N/A"
        )
    else:
        summary = ", ".join(f"{k}={v}" for k, v in units_by_type.items())
        notes.append(
            f"{total} pedagogical unit annotation(s) across "
            f"{len(units_by_type)} type(s): {summary}"
        )
    details = {
        "semantik_unit_total": total,
        "semantik_units_by_type": units_by_type,
        "distinct_unit_types": len(units_by_type),
    }
    # Report-only: informational placeholder score; never enters the composite.
    return DimensionResult("pedagogy", False, 1.0, "REPORT", details, notes)


# ---------------------------------------------------------------------------
# Per-file scoring
# ---------------------------------------------------------------------------
_DIMENSION_FNS = [
    _dim_deliver,
    _dim_heading,
    _dim_landmarks,
    _dim_cleanliness,
    _dim_image_a11y,
    _dim_math,
    _dim_language,
    _dim_pedagogy,
]


def score_html(html: str, cfg: dict[str, Any] | None = None, path: str = "<string>") -> FileResult:
    cfg = cfg or CONFIG
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    dims: dict[str, DimensionResult] = {}
    for fn in _DIMENSION_FNS:
        d = fn(soup, body, cfg)
        dims[d.name] = d

    weights = cfg["weights"]
    num = 0.0
    den = 0.0
    for name, w in weights.items():
        d = dims.get(name)
        if d is None or not d.scored:
            continue
        num += w * d.score
        den += w
    composite = (num / den) if den else 0.0
    return FileResult(path, composite, _verdict(composite, cfg), dims)


def score_file(path: Path, cfg: dict[str, Any] | None = None) -> FileResult:
    html = path.read_text(encoding="utf-8", errors="replace")
    return score_html(html, cfg, str(path))


# ---------------------------------------------------------------------------
# Aggregation across a directory
# ---------------------------------------------------------------------------
def _collect_html(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.rglob("*.html") if p.is_file())
    raise FileNotFoundError(f"--html target not found: {target}")


def aggregate(results: list[FileResult], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or CONFIG
    if not results:
        return {"file_count": 0}
    dim_names = list(results[0].dimensions.keys())
    per_dim_mean: dict[str, float] = {}
    for name in dim_names:
        vals = [r.dimensions[name].score for r in results if name in r.dimensions]
        per_dim_mean[name] = round(sum(vals) / len(vals), 4) if vals else 0.0
    composites = [r.composite for r in results]
    mean_composite = sum(composites) / len(composites)
    verdict_counts: dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in results:
        verdict_counts[r.verdict] = verdict_counts.get(r.verdict, 0) + 1
    return {
        "file_count": len(results),
        "mean_composite": round(mean_composite, 4),
        "aggregate_verdict": _verdict(mean_composite, cfg),
        "per_dimension_mean": per_dim_mean,
        "verdict_counts": verdict_counts,
        "worst_files": [
            {"path": r.path, "composite": round(r.composite, 4), "verdict": r.verdict}
            for r in sorted(results, key=lambda r: r.composite)[:5]
        ],
    }


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------
_SCORED_ORDER = ["deliver", "heading", "landmarks", "cleanliness", "image_a11y", "language"]
_REPORT_ORDER = ["math", "pedagogy"]


def render_text(results: list[FileResult], agg: dict[str, Any], baseline: bool) -> str:
    out: list[str] = []
    label = "BASELINE / GAP MEASUREMENT" if baseline else "STRUCTURE SCORECARD"
    out.append(f"=== SemantiK {label} ===")
    out.append(f"files scored: {agg.get('file_count', 0)}")
    if agg.get("file_count"):
        out.append(
            f"aggregate: mean_composite={agg['mean_composite']:.3f} "
            f"[{agg['aggregate_verdict']}]  "
            f"PASS={agg['verdict_counts'].get('PASS',0)} "
            f"WARN={agg['verdict_counts'].get('WARN',0)} "
            f"FAIL={agg['verdict_counts'].get('FAIL',0)}"
        )
        out.append("per-dimension mean (scored dims + report-only 'math'):")
        for name in _SCORED_ORDER + _REPORT_ORDER:
            if name in agg["per_dimension_mean"]:
                tag = "" if name in _SCORED_ORDER else "  (report-only)"
                out.append(f"    {name:12s} {agg['per_dimension_mean'][name]:.3f}{tag}")
    out.append("")
    for r in results:
        out.append(f"[{r.verdict}] {r.path}  composite={r.composite:.3f}")
        for name in _SCORED_ORDER + _REPORT_ORDER:
            d = r.dimensions.get(name)
            if not d:
                continue
            tag = "" if d.scored else " (report-only)"
            out.append(f"    - {name:12s} {d.score:.3f} [{d.verdict}]{tag}")
            for n in d.notes:
                out.append(f"        · {n}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Deterministic domain-agnostic structure/a11y scorecard for "
        "SemantiK-converted HTML."
    )
    ap.add_argument("--html", required=True, help="An .html file or a directory of them.")
    ap.add_argument("--json-out", default=None, help="Write the full JSON report here.")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Label this run as a baseline/gap measurement (header only).",
    )
    args = ap.parse_args(argv)

    target = Path(args.html)
    try:
        files = _collect_html(target)
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if not files:
        sys.stderr.write(f"no .html files found under {target}\n")
        return 2

    results = [score_file(p) for p in files]
    agg = aggregate(results)

    report = {
        "tool": "structure_scorecard",
        "version": "1.0",
        "baseline": bool(args.baseline),
        "config": CONFIG,
        "aggregate": agg,
        "files": [r.to_dict() for r in results],
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(render_text(results, agg, args.baseline))

    # Exit non-zero if any file FAILs (useful when wired as a gate later); a
    # pure baseline run reports the gap but never fails the process.
    if not args.baseline and agg.get("verdict_counts", {}).get("FAIL", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

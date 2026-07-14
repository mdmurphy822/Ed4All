"""SEMANTIK_AFFORDANCE_GATE — deterministic affordance-conservation audit (T0).

**Deterministic pass — no LLM call site — NO DecisionCapture obligation.**
(House convention, flagged loudly; mirrors the banner in
``SemantiK/semantik_structure/unit_coverage.py``.)

Motivation — the "gates that don't look" class
==============================================

A real chapter conversion shipped a ``{stem}.quality.json`` self-certifying
``quality_score 1.0`` / ``wcag_status: passed`` / ``certified`` over a document
carrying **0** ``<math>`` elements against 3,884 convertible LaTeX spans, **0**
``<th>`` against 28 reconstructible table topologies, and a ``<title>`` reading
``"Chapter 1 Foundations 55 ✓ Solution"``. Three distinct defect classes, none
of which ANY existing gate could see:

* **CLASS-A — DESTRUCTION.** ``math_fold.sanitize_body_latex`` deleted the
  ``| --- | --- |`` markdown separator row as "garbage" before
  ``structure_emit.parse_table`` could read it — but that row IS the pipe grid's
  topology declaration, so parse_table's ``<thead>`` / ``<th scope=col>`` branch
  was UNREACHABLE DEAD CODE on the live path (1,421 parse_table calls on a real
  render; separator present in ZERO). We were destroying the answer we already
  had.
* **CLASS-B — MISSING TRANSFORM.** 3,884 literal ``$…$`` spans shipped INTACT to
  the learner page. No renderer existed. A screen reader said "dollar backslash
  frac".
* **CLASS-C — UNWIRED REPAIR.** ``lib/textbook_title_sanitize.py`` worked
  perfectly and was wired into the EXTRACTOR only; the same run's HTML shipped
  the unsanitized running-header title.

``SEMANTIK_UNIT_COVERAGE_GATE`` was ON and green throughout — correctly. Its
universe is ``_TOKEN_RE = [^\\W_]+`` (alphanumeric TEXT TOKENS): a shipped
literal ``$\\frac{3}{4}$`` has tokens ``frac``/``3``/``4`` present → coverage
1.0 while the SEMANTICS are 0, and ``| --- |`` cells are pure punctuation → not
tokens at all → invisible by construction. That gate is not broken; it measures
a different, satisfied invariant (no text was LOST). This module is the
orthogonal axis: text can survive while its AFFORDANCE — the thing that makes it
usable by an assistive technology — is destroyed, never built, or built and left
unwired.

The design rule that makes this safe
====================================

**An evidence predicate MUST BE the emitter's own accept predicate, never an
independent heuristic racing it.** Then ``evidence`` == "convertible",
``emitted`` == "converted", and any gap is attrition BY DEFINITION — never a
false positive from a parallel estimator. So this module REUSES, never
reimplements:

* **math** — :func:`lib.semantik.latex_mathml.convert_latex_spans`, which IS the
  render's span-SELECTION seam, replayed over the render's own normalization
  chain (``_esc_text`` → ``sanitize_body_latex`` → ``wrap_bare_math`` →
  ``sanitize_math_spans``) with the document-wide ``prose_vocab``. Segmentation
  AND accept predicate are the emitter's BY CONSTRUCTION. The ``(?<!\\)``
  escaped-dollar lookbehind + :func:`~lib.semantik.math_fold._is_math_content`
  are what keep ``"costs $5 and $10"`` from counting as math (a naive
  ``\\$[^$]+\\$`` counter false-fires; these guards do not).

  This is the one predicate that got the design rule WRONG on the first landing
  and had to be repaired (task #59): it re-segmented the RAW IR text with a bare
  ``_MATH_SPAN_ANGLE_RE`` loop — a second segmenter racing the render — and so
  UNDERCOUNTED the denominator and false-fired ``over_emit`` ("+22 FABRICATED")
  on a document with ZERO fabrications. See :func:`_math_evidence`. The lesson is
  the module's own thesis: a metric computed from a parse nobody checked against
  the real thing is exactly the failure this gate exists to catch, and writing it
  a second time is how you reintroduce it.
* **table_headers** — :func:`lib.semantik.table_structure.parse_table_topology`
  (the emitter's own accept, INCLUDING its header-label admissibility gate) run
  on the PRE-destruction text, i.e. ``sanitize_body_latex(..., keep_md_sep=True)``.
  Not ``has_separator_row``: OCR garbage produces a spurious pipe-dash-pipe run
  (measured: 1 such block on a non-table medical scan), and the topology parser
  correctly refuses it — so the strong predicate is naturally domain-agnostic and
  silent off-domain.
* **sections** — the document's OWN outline declaration, harvested by the SAME
  :meth:`~lib.semantic_structure_extractor.semantic_structure_extractor.SemanticStructureExtractor._harvest_outline`
  that ``ED4ALL_STRUCTURE_OUTLINE_ANCHOR`` already uses. NOT an ``N.M`` line-shape
  regex — that was MEASURED NOISY (it matches ``7.275``, ``0.31``, ``Figure 1.3``,
  ``TRY IT :: 1.3``, and still misses ``1.3`` when fused mid-line).
* **title_purity** — :func:`lib.textbook_title_sanitize.sanitize_running_header_title`
  (the existing repair) as its own detector: if the sanitizer would CHANGE the
  emitted title, the title carries furniture the repair was built to remove and
  the seam is unwired.

NO SUBJECT-MATTER WORD LIST ANYWHERE. Every predicate is shape- or
self-declaration-based (owner doctrine: gates stay domain-agnostic; publisher
vocabulary lives in ``schemas/taxonomies/`` lexicons, never in a gate).

Honest scope (READ THIS BEFORE TRUSTING A GREEN REPORT)
======================================================

This pass audits attrition AFTER the cascade result is serialized. It does NOT
catch:

* **(a) genuine absence.** A page-raster scan has ZERO sub-page image objects —
  the figures are pixels inside the full-page raster — so the ``images`` check's
  evidence is ~0 and it CANNOT see a real 352-figure gap. It is built
  SHADOW-ONLY for that reason and must never be read as an all-clear (see
  :data:`_IMAGE_DECLARED_DROPS`).
* **(b) upstream structural destruction** unless the evidence survives as TEXT.
  A heading the arranger fused into a running-header block is already GONE by the
  time the IR is written — which is exactly why the ``sections`` predicate is
  anchored on the document's own outline/apparatus ORDINAL SPINE (a self-
  declaration that survives independently of the section-title heading) rather
  than on the emitted headings alone.

Scored against the four defects that motivated it: math + table_headers are
caught as originally proposed; sections is caught only with the outline-anchored
predicate; title is caught only after reclassification as PURITY (the count is
1 emitted vs 1 declared and perfectly conserved — no count invariant can see it).
This does NOT "catch all four on any corpus".

Contract
========

* **Gate ``SEMANTIK_AFFORDANCE_GATE`` (default OFF).** Off → the gate never runs,
  the report key is simply ABSENT from ``cascade_ir.json`` + ``quality.json``
  (byte-identical; the ``unit_coverage`` conditional-key precedent).
* **Pure / deterministic → NO ``DecisionCapture``** (flagged loudly here).
* **Pure READ pass** over artifacts that already exist. ``raw_text`` is never
  touched → ``data-dart-block-id`` / ``dart:{slug}#{block_id}`` sourceIds are
  byte-stable. Zero sourceId / byte impact.
* **Verdicts come from the PERSISTED artifacts only** (IR + HTML). The in-memory
  emitter self-report is corroboration, never a verdict basis — so an out-of-band
  audit of a shipped corpus grades a document identically to the wired seam.
* **NO MEASUREMENT IS NOT A PASS.** ``document_passed`` is tri-state
  (``True`` / ``False`` / ``None``); a check that could not look emits a
  ``not_evaluated`` issue, poisons the verdict to ``None``, and DE-CERTIFIES the
  quality sidecar. A gate that cannot evaluate never contributes a green — that
  is the founding sin this whole gate exists to kill, and it is not permitted one
  layer up either.

Both-sided by construction
==========================

Every count check flags BOTH directions: a shortfall (attrition) AND an over-emit
(fabrication). ``latex_to_mathml(" and ")`` really does return
``<mi>a</mi><mi>n</mi><mi>d</mi>``, so a prose span mis-wrapped as math becomes a
``<math>`` element that reads as gibberish — and per the standing owner framing, a
``<th>`` (or a ``<math>``) IN THE WRONG PLACE IS WORSE THAN EMITTING NOTHING. The
over-emit arms are COUNT arms against the emitter's own accept predicate, never
shape rules: a run of single-letter ``<mi>`` also matches a legitimate ``$abc$``,
so a shape-based fabrication detector would false-fire on correct math.

False positives (the ones that were actually hunted, on real files)
==================================================================

* **``$`` as CURRENCY.** ``"the car cost $24,493. He paid with a check … $152"``
  is prose spanning two dollar signs. Handled by REUSING ``math_fold``'s own
  ``_is_math_content`` / ``_is_prose_math_span`` (majority-prose test) rather than
  a naive ``\\$[^$]+\\$`` — and by scanning PLAIN TEXT, not markup (against markup
  the ``</p>`` / ``<div`` tokens inside the span are not prose words, which flips
  the majority test and reports the currency prose as shipped LaTeX; measured on
  the vendor gold: 5 false residue spans → 0).
* **A table with NO headers BY DESIGN.** Evidence is
  ``parse_table_topology`` — the emitter's own accept, INCLUDING its header-label
  admissibility gate — not "a table exists" and not ``has_separator_row``. A
  headerless grid declares no topology, so evidence is 0 and the check is silent.
  An OCR garble that emits a spurious pipe-dash run is REFUSED by the same parser.
* **A non-math / non-table document.** Both evidence predicates are the emitters'
  own accepts, so they are ~0 off-domain: measured completely SILENT (0 issues) on
  a non-math medical scan, in both the OCR and clean-text lanes.
* **A legitimate title embedding a year** (``"The 1984 Report"``) can trip the
  folio-token arm — which is precisely why ``title_purity`` is WARNING severity
  and never critical.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "SEMANTIK_AFFORDANCE_GATE_ENV",
    "SEMANTIK_SECTION_RECALL_MIN_ENV",
    "resolve_affordance_gate_mode",
    "resolve_section_recall_min",
    "audit_affordances",
    "run_affordance_gate",
]

SEMANTIK_AFFORDANCE_GATE_ENV = "SEMANTIK_AFFORDANCE_GATE"
SEMANTIK_SECTION_RECALL_MIN_ENV = "SEMANTIK_AFFORDANCE_SECTION_RECALL_MIN"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# TODO(calibration): section-recall floors. These are CALIBRATION constants, not
# corpus targets — a document's declared ordinal spine vs the headings that
# anchor it. 0.80 is the warn floor; a spine of >= _SECTION_MIN_DECLARED with
# ZERO anchored headings is the hard (critical) floor: the document declares a
# structure and the render delivers none of it.
_DEFAULT_SECTION_RECALL_MIN = 0.80
_SECTION_MIN_DECLARED = 3

_SCHEMA = "semantik_affordance_conservation/v1"

# Images: evidence the pipeline DELIBERATELY destroys, so a low image count is
# NOT proof of attrition. Named here so a green images row can never read as an
# all-clear (see the module docstring, honest-scope (a)).
_IMAGE_DECLARED_DROPS = (
    "adapter._strip_markdown_images: VLM-fabricated ![](url) destroyed by design "
    "(the URL is hallucinated) — its evidence is unrecoverable here",
    "adapter._strip_literal_img_tags: fabricated raw <img src=https://…> tag text "
    "destroyed by design",
    "page-raster scan: the PDF carries NO sub-page image objects (figures are "
    "pixels inside the full-page raster), so evidence ~= 0 and a REAL figure gap "
    "is INVISIBLE to this check",
)

# Region kinds the pipeline DESTROYS BY DESIGN — their text can NEVER reach the
# learner page, so their spans were never conservable evidence. Excluding them is
# the same principle :data:`_IMAGE_DECLARED_DROPS` already encodes for images: a
# DECLARED intentional drop is not attrition, and counting it as evidence
# manufactures a phantom shortfall.
#
# This NARROWS the denominator, so it can never LAUNDER an over-emit — it makes
# the fabrication arm STRICTLY HARDER to satisfy, never easier. (Measured on a
# real chapter: ``metadata_drop`` regions carry exactly 40 convertible spans, and
# the emitter's own self-report converts 3,946 while only 3,906 <math> ship —
# a difference of EXACTLY those 40. The arithmetic closes.)
_RENDER_DROPPED_REGION_KINDS = frozenset({"metadata_drop"})

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MATH_ELEMENT_RE = re.compile(r"<math\b.*?</math\s*>", re.DOTALL | re.IGNORECASE)
_MATH_OPEN_RE = re.compile(r"<math\b", re.IGNORECASE)
_THEAD_OPEN_RE = re.compile(r"<thead\b", re.IGNORECASE)
_TH_OPEN_RE = re.compile(r"<th\b", re.IGNORECASE)
_TABLE_OPEN_RE = re.compile(r"<table\b", re.IGNORECASE)
_IMG_OPEN_RE = re.compile(r"<img\b", re.IGNORECASE)
_TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title\s*>", re.DOTALL | re.IGNORECASE)
_H_RE = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>", re.DOTALL | re.IGNORECASE)
# A LaTeX control word surviving OUTSIDE any <math> element / delimited span.
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]{2,}")


# ---------------------------------------------------------------------------
# Flag resolution (parse-with-fallback; read at CALL time).
# ---------------------------------------------------------------------------
def resolve_affordance_gate_mode() -> bool:
    """Whether the affordance-conservation gate may run (default OFF).

    Parse-with-fallback truthy set; unset / blank / falsey / garbage → off
    (byte-identical — the gate never runs and no report key is emitted).
    """
    return (os.environ.get(SEMANTIK_AFFORDANCE_GATE_ENV) or "").strip().lower() in _TRUTHY


def resolve_section_recall_min() -> float:
    """Outline-anchored section-recall floor below which the ``sections`` check
    WARNS (default ``0.80``). Float parse-with-fallback in ``[0, 1]``."""
    raw = os.environ.get(SEMANTIK_SECTION_RECALL_MIN_ENV)
    if raw is None or not str(raw).strip():
        return _DEFAULT_SECTION_RECALL_MIN
    try:
        val = float(str(raw).strip())
    except (TypeError, ValueError):
        return _DEFAULT_SECTION_RECALL_MIN
    if val != val or val in (float("inf"), float("-inf")) or val < 0.0 or val > 1.0:
        return _DEFAULT_SECTION_RECALL_MIN
    return val


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _iter_region_texts(cascade_ir_doc: Any) -> Iterator[Tuple[Any, str]]:
    """Yield ``(region_kind, text)`` for every region in the IR.

    Prefers ``repaired_text`` over ``raw_text`` — the SAME precedence the render
    seam (``cascade_ir._resolve_repaired``) uses, so the evidence is computed on
    the text the emitter actually sees.
    """
    if not isinstance(cascade_ir_doc, dict):
        return
    regions = cascade_ir_doc.get("region_provenance") or []
    if not isinstance(regions, list):
        return
    for entry in regions:
        if not isinstance(entry, dict):
            continue
        text = entry.get("repaired_text") or entry.get("raw_text") or ""
        yield entry.get("region_kind"), str(text)


def _strip_math_elements(html: str) -> str:
    """Drop every ``<math>…</math>`` element.

    LOAD-BEARING for the residue check: a CORRECTLY converted ``<math>`` carries
    the original LaTeX twice on purpose — as ``alttext`` (WCAG 1.1.1) and inside
    ``<annotation encoding="application/x-tex">`` (lossless round-trip). Scanning
    for residue without removing those first would report a perfectly-converted
    document as full of source notation.
    """
    return _MATH_ELEMENT_RE.sub(" ", html or "")


def _plain_text(html: str) -> str:
    """Tags removed, entities unescaped — the text a reader actually sees."""
    import html as _html_mod  # noqa: PLC0415

    return _html_mod.unescape(_HTML_TAG_RE.sub(" ", html or ""))


# ---------------------------------------------------------------------------
# Check: math (CLASS-B detector).
# ---------------------------------------------------------------------------
def _math_evidence(texts: List[str]) -> Tuple[int, int]:
    """``(spans, convertible)`` over the RENDER'S OWN span universe.

    **This function must not segment spans itself.** The original implementation
    did — it ran :data:`~lib.semantik.math_fold._MATH_SPAN_ANGLE_RE` straight over
    the RAW IR region text and adjudicated each hit with a bare
    ``latex_to_mathml(body)``. That is a SECOND SEGMENTER racing the render, and
    it FALSE-FIRED the ``over_emit`` (fabrication) arm on a clean document:

    * The render never converts raw text. It converts the ASSEMBLED BLOCK HTML,
      after ``sanitize_body_latex`` + ``wrap_bare_math`` (which WRAPS bare,
      un-delimited math runs into ``$…$`` — spans the raw scan cannot see at all)
      and ``sanitize_math_spans`` (which REPAIRS span content, turning a span the
      grammar would decline into one it accepts). So the raw-text denominator
      systematically UNDERCOUNTED.
    * The raw scan also dropped the ``prose_vocab`` argument, so it adjudicated
      with a DIFFERENT accept predicate than the render's (task #57 rejects a
      prose fragment like ``$ and $`` only when the document vocabulary is
      supplied).

    Measured on a real 1,161-region chapter: raw scan 3,884 "convertible" vs
    3,906 emitted ``<math>`` → a phantom "+22 FABRICATED" warning over a document
    whose only pure letter-run spans were the LEGITIMATE variable products ``ab``
    and ``cd`` (``<mi>a</mi><mi>b</mi>`` is the CORRECT MathML for the product
    ``ab``). Nothing was fabricated. The gate was computing a metric from a parse
    it had never checked against what the renderer actually does — the SAME sin,
    mirrored, that this whole module exists to kill.

    So the evidence side now REPLAYS THE RENDER'S OWN CHAIN, reusing every
    function verbatim (never re-deriving a boundary):

    ``_esc_text`` → ``sanitize_body_latex(html=True)`` → ``wrap_bare_math(html=True)``
    → ``sanitize_math_spans`` → :func:`~lib.semantik.latex_mathml.convert_latex_spans`

    — the last of which IS the render's span-SELECTION seam (``adapter.
    _latex_to_mathml`` calls exactly it, with exactly this document-wide
    ``prose_vocab``). Segmentation and accept predicate are therefore IDENTICAL to
    the emitter's BY CONSTRUCTION rather than by agreement, which is the module's
    founding design rule applied to itself. Corroboration that the parity is real:
    on that same chapter the replay's DECLINED count is 4 — the emitter's own
    self-report is also exactly 4.

    ``convertible`` is what that seam CONVERTS; ``spans`` is what it SEES
    (converted + declined).
    """
    from lib.semantik.adapter import _esc_text  # noqa: PLC0415
    from lib.semantik.latex_mathml import convert_latex_spans  # noqa: PLC0415
    from lib.semantik.math_fold import (  # noqa: PLC0415
        build_prose_vocabulary,
        sanitize_body_latex,
        sanitize_math_spans,
        wrap_bare_math,
    )

    # The DOCUMENT's ordinary-prose vocabulary — harvested exactly as
    # ``adapter._latex_to_mathml`` harvests it: from the post-fold PLAIN-text
    # (``block.raw_text``) lane, with every math span removed. Without it the
    # accept predicate is not the render's (see the docstring).
    plain = [
        wrap_bare_math(sanitize_body_latex(t, html=False), html=False) for t in texts
    ]
    prose_vocab = build_prose_vocabulary(plain)

    spans = 0
    convertible = 0
    for text in texts:
        if not text:
            continue
        # Reconstruct the block HTML the render converts (adapter's own text-block
        # build), then the span-content repair pass, then the SELECTION seam.
        block_html = wrap_bare_math(
            sanitize_body_latex(f"<p>{_esc_text(text)}</p>", html=True), html=True
        )
        block_html = sanitize_math_spans(block_html)
        _out, converted, declined = convert_latex_spans(
            block_html, prose_vocab=prose_vocab
        )
        convertible += converted
        spans += converted + declined
    return spans, convertible


def _check_math(texts: List[str], html: str, emitter: Dict[str, Any]) -> Dict[str, Any]:
    spans, convertible = _math_evidence(texts)
    emitted = len(_MATH_OPEN_RE.findall(html))
    converted = emitter.get("math_spans_converted")
    declined = emitter.get("math_spans_declined")
    ran = isinstance(converted, int) and isinstance(declined, int) and (
        converted + declined
    ) > 0

    check: Dict[str, Any] = {
        "evidence_spans_total": spans,
        "evidence_convertible": convertible,
        "emitted_math_elements": emitted,
        "emitter_converted": converted,
        "emitter_declined": declined,
        "verdict": "pass",
        "severity": None,
        "message": "",
    }
    if convertible == 0:
        check["message"] = "no convertible math evidence — nothing to conserve"
        if emitted > 0:
            # FABRICATION, the inverse failure — and the axis where it is KNOWN
            # to occur: `latex_to_mathml(" and ")` returns
            # `<mi>a</mi><mi>n</mi><mi>d</mi>`, so a prose span mis-wrapped as
            # math becomes a <math> element that reads as gibberish to a screen
            # reader. Symmetric with _check_table_headers' over_emit arm: a
            # <math> in the wrong place is WORSE than emitting nothing.
            check["verdict"] = "over_emit"
            check["severity"] = "warning"
            check["message"] = (
                f"{emitted} <math> element(s) emitted with ZERO convertible math "
                f"evidence in the source — every one is FABRICATED (the emitter's "
                f"own accept predicate admits none of this document's spans)"
            )
        return check
    if emitted == 0:
        # The transform produced NOTHING over nonzero evidence. If the emitter
        # never reported (pass absent / flag off) this is CLASS-B: the pipeline
        # is shipping source notation. If it DID report, it declined everything —
        # a documented refusal, loud but not a fabrication risk.
        if ran:
            check["verdict"] = "refused"
            check["severity"] = "warning"
            check["message"] = (
                f"the math emitter RAN and declined every span "
                f"({declined} declined / {converted} converted) over "
                f"{convertible} convertible span(s) — no <math> shipped"
            )
        else:
            check["verdict"] = "fail"
            check["severity"] = "critical"
            check["message"] = (
                f"{convertible} convertible LaTeX span(s) in the source and ZERO "
                f"<math> elements emitted — the LaTeX->MathML transform did not "
                f"run (SEMANTIK_LATEX_MATHML). Literal source notation is being "
                f"read aloud to the learner (WCAG 1.3.1)."
            )
        return check
    if emitted < convertible:
        check["verdict"] = "shortfall"
        check["severity"] = "warning"
        check["message"] = (
            f"{emitted} <math> emitted against {convertible} convertible span(s) "
            f"— {convertible - emitted} span(s) unaccounted for"
        )
    elif emitted > convertible:
        # The OTHER side of the same invariant (D2). The math lane is KNOWN to
        # fabricate on prose mis-wrapped as math, and the emitter's own accept
        # predicate is the denominator — so an excess is, by the same definition
        # that makes a shortfall attrition, an OVER-EMIT. Deliberately a COUNT
        # arm, not a shape rule: a run of single-letter <mi> also matches a
        # legitimate `$abc$`, so a shape-based fabrication detector would false-
        # fire on correct math. A count arm needs no shape rule, and on a
        # conserved document (emitted == convertible) it is silent.
        check["verdict"] = "over_emit"
        check["severity"] = "warning"
        check["message"] = (
            f"{emitted} <math> emitted against only {convertible} convertible "
            f"span(s) — {emitted - convertible} FABRICATED element(s): the "
            f"emitter produced markup for spans its own accept predicate does "
            f"not admit (prose mis-wrapped as math reads as gibberish to a "
            f"screen reader)"
        )
    return check


# ---------------------------------------------------------------------------
# Check: table_headers (CLASS-A detector).
# ---------------------------------------------------------------------------
def _table_evidence(texts: List[str]) -> int:
    """Count of BLOCKS whose table topology the EMITTER'S OWN PARSER accepts.

    Runs :func:`lib.semantik.table_structure.parse_table_topology` — including its
    header-label admissibility gate — on the PRE-destruction text (the
    ``keep_md_sep=True`` sanitation the table lane itself stashes). Per BLOCK, not
    row-count vs cell-count, so this also guards the measured 65%-false-positive
    ``<th>`` failure mode (the admissibility gate is inside the predicate).

    Deliberately NOT ``has_separator_row``: an OCR garble can emit a spurious
    pipe-dash-pipe run, and the topology parser refuses it — which is what makes
    this evidence naturally silent on a non-table corpus.
    """
    from lib.semantik.math_fold import sanitize_body_latex, wrap_bare_math  # noqa: PLC0415
    from lib.semantik.table_structure import (  # noqa: PLC0415
        has_separator_row,
        parse_table_topology,
    )

    count = 0
    for text in texts:
        if not text or not has_separator_row(text):
            continue  # cheap pre-check, exactly the adapter's
        src = wrap_bare_math(
            sanitize_body_latex(text, html=False, keep_md_sep=True), html=False
        )
        if parse_table_topology(src) is not None:
            count += 1
    return count


def _check_table_headers(
    texts: List[str], html: str, emitter: Dict[str, Any]
) -> Dict[str, Any]:
    evidence = _table_evidence(texts)
    emitted = len(_THEAD_OPEN_RE.findall(html))
    accepted = emitter.get("tables_accepted")
    refused = emitter.get("tables_refused")
    ran = isinstance(accepted, int) and isinstance(refused, int) and (
        accepted + refused
    ) > 0

    check: Dict[str, Any] = {
        "evidence_reconstructible_tables": evidence,
        "emitted_thead_blocks": emitted,
        "emitted_th_cells": len(_TH_OPEN_RE.findall(html)),
        "emitted_tables": len(_TABLE_OPEN_RE.findall(html)),
        "emitter_accepted": accepted,
        "emitter_refused": refused,
        "verdict": "pass",
        "severity": None,
        "message": "",
    }
    if evidence == 0:
        check["message"] = "no reconstructible table topology declared — nothing to conserve"
        if emitted > 0:
            # Headers emitted with NO admissible evidence: the inverse failure —
            # a <th> in the wrong place actively MISLEADS a screen-reader user.
            check["verdict"] = "over_emit"
            check["severity"] = "warning"
            check["message"] = (
                f"{emitted} <thead> emitted with ZERO admissible header evidence "
                f"— a fabricated column header misleads assistive tech"
            )
        return check
    if emitted == 0:
        if ran:
            check["verdict"] = "refused"
            check["severity"] = "warning"
            check["message"] = (
                f"the table lane RAN and refused every declared table "
                f"({refused} refused / {accepted} accepted) over {evidence} "
                f"reconstructible topology declaration(s)"
            )
        else:
            check["verdict"] = "fail"
            check["severity"] = "critical"
            check["message"] = (
                f"{evidence} block(s) declare a reconstructible table topology "
                f"(a '| --- |' separator row the source EMITTED) and ZERO <thead> "
                f"shipped — the cell-topology lane did not run "
                f"(SEMANTIK_TABLE_STRUCTURE). Every table is a headerless <td> "
                f"soup a screen reader cannot navigate (WCAG 1.3.1)."
            )
        return check
    if emitted > evidence:
        check["verdict"] = "over_emit"
        check["severity"] = "warning"
        check["message"] = (
            f"{emitted} <thead> emitted against only {evidence} admissible "
            f"topology declaration(s) — possible fabricated header"
        )
    elif emitted < evidence:
        # A gap here is NOT necessarily attrition: `emit_structured_table` runs
        # strictly MORE checks than `parse_table_topology` (text conservation +
        # the cascade's own WCAG H43 gate), so it can honestly refuse a topology
        # the parser admits. The self-report says which.
        check["verdict"] = "shortfall"
        check["severity"] = "warning"
        check["message"] = (
            f"{emitted} <thead> emitted against {evidence} reconstructible "
            f"topology declaration(s)"
            + (
                f" — the lane accepted {accepted} and refused {refused} of "
                f"{emitter.get('tables_attempted')} attempted (a refusal is the "
                f"H43 gate / text-conservation check failing closed, not attrition)"
                if ran
                else " — the cell-topology lane reported nothing"
            )
        )
    return check


# ---------------------------------------------------------------------------
# Check: sections (outline-anchored).
# ---------------------------------------------------------------------------
def _check_sections(html: str, recall_min: float) -> Dict[str, Any]:
    """Declared ordinal spine vs the emitted headings that anchor it.

    The declaration comes from the extractor's OWN
    ``_harvest_outline`` (the ``ED4ALL_STRUCTURE_OUTLINE_ANCHOR`` harvest) — an
    ordinal is admitted only with STRUCTURAL evidence (it opens a heading element
    or forms an ``N.M Exercises`` apparatus banner), so the spine survives even
    when the section-TITLE heading was destroyed. That is what keeps this check
    from being circular: the apparatus banners declare the spine, the section
    titles are what go missing.
    """
    check: Dict[str, Any] = {
        "declared_ordinals": 0,
        "anchored_ordinals": 0,
        "recall": None,
        "recall_min": recall_min,
        "missing": [],
        "verdict": "not_evaluated",
        "severity": None,
        "message": "",
    }
    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415

        from lib.semantic_structure_extractor.semantic_structure_extractor import (  # noqa: PLC0415
            SemanticStructureExtractor,
        )
    except ImportError as exc:  # pragma: no cover - bs4 is a hard dep in practice
        check["message"] = f"outline harvest unavailable ({exc}) — NOT EVALUATED"
        return check

    soup = BeautifulSoup(html or "", "html.parser")
    harvester = SemanticStructureExtractor.__new__(SemanticStructureExtractor)
    entries, _outline_articles, _sources = harvester._harvest_outline(soup)
    declared = dict(entries or {})
    check["declared_ordinals"] = len(declared)
    if len(declared) < _SECTION_MIN_DECLARED:
        check["verdict"] = "pass"
        check["message"] = (
            f"{len(declared)} declared ordinal(s) — below the {_SECTION_MIN_DECLARED} "
            f"floor needed to judge a spine"
        )
        return check

    headings = [
        _norm(_plain_text(m.group(2))).lower() for m in _H_RE.finditer(html or "")
    ]
    anchored: List[str] = []
    for ordinal, title in sorted(declared.items()):
        hit = False
        for text in headings:
            # (1) the heading CARRIES the ordinal (e.g. "1.3 Add and Subtract…")
            if text.startswith(ordinal) and not text[len(ordinal) : len(ordinal) + 1].isdigit():
                hit = True
                break
            # (2) the heading IS the outline-donated title for that ordinal
            norm_title = _norm(title).lower()
            if len(norm_title) >= 6 and norm_title in text:
                hit = True
                break
        if hit:
            anchored.append(ordinal)
    check["anchored_ordinals"] = len(anchored)
    missing = sorted(set(declared) - set(anchored))
    check["missing"] = missing[:20]
    recall = len(anchored) / len(declared)
    check["recall"] = round(recall, 4)

    if len(anchored) == 0:
        check["verdict"] = "fail"
        check["severity"] = "critical"
        check["message"] = (
            f"the document declares a {len(declared)}-ordinal section spine and "
            f"NOT ONE emitted heading anchors it — the declared structure was "
            f"destroyed before render"
        )
    elif recall < recall_min:
        check["verdict"] = "shortfall"
        check["severity"] = "warning"
        check["message"] = (
            f"section recall {recall:.2f} < {recall_min:.2f} — "
            f"{len(anchored)}/{len(declared)} declared ordinals have an anchoring "
            f"heading; missing {missing[:8]}"
        )
    else:
        check["verdict"] = "pass"
        check["message"] = f"section recall {recall:.2f} ({len(anchored)}/{len(declared)})"
    return check


# ---------------------------------------------------------------------------
# Check: residue (CLASS-B detector, cheap + domain-agnostic).
# ---------------------------------------------------------------------------
def _check_residue(html: str, emitter: Dict[str, Any]) -> Dict[str, Any]:
    """Source notation that must NEVER survive rendering.

    This is exactly the check ``quality.json`` should have been making instead of
    self-certifying 1.0. Computed on the html with ``<math>`` elements REMOVED —
    a correct conversion legitimately carries the original LaTeX inside
    ``alttext`` + ``<annotation>``, and counting those would report a perfect
    document as broken.

    The delimited-span arm is the load-bearing one and is FP-tested: it reuses
    :func:`~lib.semantik.math_fold._is_math_content`, so ``"costs $5 and $10"``
    and a headerless markdown grid both yield ZERO residue.

    The scan runs on PLAIN TEXT (tags stripped, entities unescaped), NOT on the
    raw markup — this is load-bearing, and getting it wrong produced a real false
    positive during validation. ``_is_math_content`` delegates to
    ``_is_prose_math_span``, whose majority-prose test counts only PURELY
    ALPHABETIC tokens; run against markup, the ``</p>`` / ``<div`` / ``class="…"``
    tokens inside a span are not prose words, so a span spanning two currency
    amounts across ordinary prose (``"$24,493. He paid for the car with a
    check.</p></div>… $152"``) flips from prose to "math" and gets reported as
    shipped LaTeX. ``math_fold``'s own pipeline stashes tags out
    (``_BARE_TAG_RE``) before ever applying this predicate; so does this. Measured:
    the vendor gold went from 5 false residue spans to 0.

    **A DECLINED span is not attrition — and the residue itself says which is
    which.** ``convert_latex_spans`` is fail-soft by contract: a span outside the
    deterministic grammar (an ``array`` environment, an unknown control sequence,
    a mis-wrapped prose blob) is left BYTE-FOR-BYTE verbatim rather than
    half-converted, so it NECESSARILY shows up as residue. Without a way to tell
    those apart, a correct render fails its own gate over the ``array``
    environments the grammar honestly refuses.

    So each residue span is ADJUDICATED INDIVIDUALLY by the emitter's own accept
    predicate — :func:`~lib.semantik.latex_mathml.latex_to_mathml`, the exact
    function the render pass calls:

    * **convertible residue** → the emitter's OWN predicate says it COULD have
      rendered this span, and it shipped as literal ``$…$`` text anyway. That is
      attrition BY DEFINITION (the design rule), so it is **CRITICAL**.
    * **non-convertible residue** → outside the deterministic grammar; shipping
      it verbatim is the documented fail-soft refusal → **warning**.

    This is deliberately span-wise and NOT a count compare against the emitter's
    declared decline total: ``len(residue) <= declined`` would let an unrelated
    decline LAUNDER a genuinely-dropped span (four residue spans "accounted for"
    by four declines need not be the SAME four). It also means the verdict is
    derived ENTIRELY from the two persisted artifacts — the in-memory
    ``emitter_report`` is corroboration, never the basis — so an out-of-band audit
    of a persisted corpus (``audit_affordances(ir, html)``, the natural operator
    path) reaches the same verdict as the wired seam.

    The message NEVER claims "the transform never ran" over a document that holds
    ``<math>``: ``emitted_math`` is dispositive proof it ran, and it is read
    straight out of the HTML being graded.
    """
    from lib.semantik.math_fold import (  # noqa: PLC0415
        _MATH_SPAN_ANGLE_RE,
        _MD_SEP_ROW_RE,
        _is_math_content,
    )
    from lib.semantik.latex_mathml import _split_span, latex_to_mathml  # noqa: PLC0415

    text = _plain_text(_strip_math_elements(html or ""))

    convertible_spans: List[str] = []
    declined_spans: List[str] = []
    for match in _MATH_SPAN_ANGLE_RE.finditer(text):
        body, display = _split_span(match.group(0))
        if not _is_math_content(body):
            continue
        if latex_to_mathml(body, display=display) is not None:
            convertible_spans.append(match.group(0))
        else:
            declined_spans.append(match.group(0))
    math_spans = convertible_spans + declined_spans
    md_sep = len(_MD_SEP_ROW_RE.findall(text))
    latex_cmds = len(_LATEX_CMD_RE.findall(text))

    # Dispositive, artifact-derived proof the transform ran: a <math> element in
    # the HTML we are grading. (The self-report is corroboration only — it is
    # in-memory BY DESIGN, so it is absent on every out-of-band audit.)
    emitted_math = len(_MATH_OPEN_RE.findall(html or ""))
    converted = emitter.get("math_spans_converted")
    declined = emitter.get("math_spans_declined")
    reported = isinstance(converted, int) and isinstance(declined, int) and (
        converted + declined
    ) > 0
    transform_ran = emitted_math > 0 or reported

    check: Dict[str, Any] = {
        "math_source_spans": len(math_spans),
        "residue_convertible": len(convertible_spans),
        "residue_declined": len(declined_spans),
        "emitted_math_elements": emitted_math,
        "transform_ran": transform_ran,
        "markdown_separator_rows": md_sep,
        "latex_control_words": latex_cmds,
        "emitter_declined": declined,
        "samples": [s[:60] for s in math_spans[:5]],
        "verdict": "pass",
        "severity": None,
        "message": "no source notation survived rendering",
    }
    if convertible_spans:
        check["verdict"] = "fail"
        check["severity"] = "critical"
        check["message"] = (
            f"{len(convertible_spans)} CONVERTIBLE LaTeX math span(s) SHIPPED as "
            f"literal text to the learner page (outside any <math>) — the emitter's "
            f"own accept predicate admits every one of them, so this is attrition, "
            f"not a refusal"
            + (
                f"; the LaTeX->MathML transform DID run ({emitted_math} <math> "
                f"emitted), so it dropped these on the floor"
                if transform_ran
                else "; ZERO <math> in the document — the transform never ran "
                "(SEMANTIK_LATEX_MATHML)"
            )
            + f". A screen reader reads these aloud as source notation (WCAG "
            f"1.3.1) — e.g. {[s[:60] for s in convertible_spans[:2]]}"
        )
    elif declined_spans:
        check["verdict"] = "declined_verbatim"
        check["severity"] = "warning"
        check["message"] = (
            f"{len(declined_spans)} LaTeX span(s) shipped verbatim — every one is "
            f"OUTSIDE the deterministic grammar (the emitter's own predicate "
            f"declines them), so this is the documented fail-soft refusal, never a "
            f"mangled half-conversion: {[s[:60] for s in declined_spans[:2]]}"
        )
    elif md_sep or latex_cmds:
        check["verdict"] = "shortfall"
        check["severity"] = "warning"
        check["message"] = (
            f"source notation residue: {md_sep} markdown separator row(s), "
            f"{latex_cmds} bare LaTeX control word(s)"
        )
    return check


# ---------------------------------------------------------------------------
# Check: title_purity (CLASS-C detector).
# ---------------------------------------------------------------------------
def _check_title_purity(html: str, sibling_title: Optional[str]) -> Dict[str, Any]:
    """PURITY, not conservation.

    The title count is 1 declared vs 1 emitted and perfectly CONSERVED, so no
    count invariant can see this defect. The signal is that the emitted title
    still carries the furniture an EXISTING repair
    (``lib/textbook_title_sanitize.py``) was built to remove — i.e. the repair is
    unwired at this seam.

    Two shape arms, both domain-agnostic:

    * **A (repair-delta)** — the sanitizer would CHANGE the title.
    * **B (folio token)** — a standalone 2-4 digit page-number token survives in
      the title (reusing the sanitizer's own ``_PAGE_NUM_RE``). Arm B exists
      because arm A's ``_MAX_TAIL_WORDS`` guard deliberately refuses to strip a
      LONG furniture tail, so it misses e.g. ``"Chapter 1 Foundations 73 73 Be
      careful to get a and b in the right order!"``.

    WARNING severity, not critical: arm B has a plausible false-positive profile
    on a legitimate title that embeds a year (``"The 1984 Report"``), and a false
    CRITICAL zeroes ``quality_score``. Loud, not fatal — the operator escalates by
    turning ``SEMANTIK_TITLE_SANITIZE`` on.

    A ``sibling_title`` (e.g. the ``textbook_structure.json`` chapter
    ``headingText`` emitted by the SAME run) is compared when supplied — a
    disagreement between two emitted artifacts is an unwired seam by
    construction. It is ``not_evaluated`` at the conversion seam, where the
    sibling does not exist yet.
    """
    from lib.textbook_title_sanitize import (  # noqa: PLC0415
        _PAGE_NUM_RE,
        sanitize_running_header_title,
    )

    match = _TITLE_RE.search(html or "")
    title = _norm(_plain_text(match.group(1))) if match else ""
    check: Dict[str, Any] = {
        "title": title,
        "sanitized": None,
        "repair_delta": False,
        "folio_token": False,
        "sibling_title": sibling_title,
        "sibling_agrees": None,
        "verdict": "pass",
        "severity": None,
        "message": "title is clean",
    }
    if not title:
        check["verdict"] = "not_evaluated"
        check["message"] = "no <title> in the emitted document"
        return check

    sanitized = sanitize_running_header_title(title)
    check["sanitized"] = sanitized
    check["repair_delta"] = sanitized != title
    check["folio_token"] = bool(_PAGE_NUM_RE.search(title))

    if sibling_title:
        check["sibling_agrees"] = _norm(sibling_title).lower() == title.lower()

    reasons: List[str] = []
    if check["repair_delta"]:
        reasons.append(
            f"the running-header sanitizer would rewrite it to {sanitized!r} "
            f"(SEMANTIK_TITLE_SANITIZE is unwired at this seam)"
        )
    if check["folio_token"]:
        reasons.append("a standalone page-number (folio) token survives in the title")
    if check["sibling_agrees"] is False:
        reasons.append(
            f"it DISAGREES with the sibling artifact title {sibling_title!r} — two "
            f"emitted artifacts of the same run disagree, so a seam is unwired"
        )

    if reasons:
        check["verdict"] = "impure"
        check["severity"] = "warning"
        check["message"] = f"emitted <title> {title!r} carries furniture: " + "; ".join(
            reasons
        )
    return check


# ---------------------------------------------------------------------------
# Check: images (SHADOW-ONLY — see the honest-scope note).
# ---------------------------------------------------------------------------
def _check_images(cascade_ir_doc: Any, html: str) -> Dict[str, Any]:
    """Figure regions the extractor DID see vs the ``<img>`` elements shipped.

    ASYMMETRIC BY CONSTRUCTION, and the asymmetry is the honest part:

    * **evidence > 0 and ZERO ``<img>`` emitted → FIRES.** Every figure the
      extractor actually saw was dropped between IR and render. That is a
      MEASURED destruction (CLASS-A) and there is no reason to stay quiet about
      it — the pre-fix ch01 carries exactly this shape (1 figure region, 0
      ``<img>``).
    * **evidence == 0 → SHADOW. NEVER an all-clear.** This is the honest limit:
      a page-raster scan has NO sub-page image objects (the figures are pixels
      inside a full-page raster), so the evidence is ~0 and a REAL 352-figure gap
      is INVISIBLE here. A green row means "nothing to compare", NOT "no figures
      are missing" (:data:`_IMAGE_DECLARED_DROPS`).

    WARNING, not critical — deliberately, and NOT for squeamishness: unlike
    ``math`` / ``table_headers``, this evidence predicate is NOT the emitter's own
    accept predicate (a ``region_kind == "figure"`` region is an extractor label,
    not a render-time "I could have emitted this"). The module's design rule says
    only an emitter-accept predicate earns a CRITICAL, because only then is a gap
    attrition BY DEFINITION. Two intentional-drop paths
    (``_strip_markdown_images`` / ``_strip_literal_img_tags``, which destroy
    VLM-FABRICATED image URLs on purpose) also sit on this seam, so a critical
    here could be a fabricated critical on a hallucinated-image corpus — the
    exact D1 sin.
    """
    figure_regions = 0
    if isinstance(cascade_ir_doc, dict):
        for entry in cascade_ir_doc.get("region_provenance") or []:
            if isinstance(entry, dict) and entry.get("region_kind") == "figure":
                figure_regions += 1
    emitted = len(_IMG_OPEN_RE.findall(html or ""))

    check: Dict[str, Any] = {
        "evidence_figure_regions": figure_regions,
        "emitted_img_elements": emitted,
        "verdict": "shadow",
        "severity": None,
        "declared_intentional_drops": list(_IMAGE_DECLARED_DROPS),
        "message": (
            "SHADOW — no figure-region evidence to compare against. This is NOT "
            "an all-clear: on a page-raster scan the evidence is ~0 by "
            "construction and a REAL figure gap is invisible here"
        ),
    }
    if figure_regions and emitted == 0:
        check["verdict"] = "fail"
        check["severity"] = "warning"
        check["message"] = (
            f"the extractor saw {figure_regions} figure region(s) and ZERO <img> "
            f"shipped — every figure it DID see was dropped before render (a "
            f"learner gets no image, and no alt text to replace it). NOTE: this "
            f"is a FLOOR, not a census — the real figure gap on a page-raster "
            f"scan is larger and invisible to this check"
        )
    elif figure_regions:
        check["verdict"] = "pass"
        check["message"] = (
            f"{emitted} <img> emitted against {figure_regions} figure region(s) "
            f"— NOT an all-clear (evidence is a floor, not a census)"
        )
    return check


# ---------------------------------------------------------------------------
# Public: audit + gate.
# ---------------------------------------------------------------------------
def audit_affordances(
    cascade_ir_doc: Any,
    html: str,
    *,
    emitter_report: Optional[Dict[str, Any]] = None,
    sibling_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministic affordance-conservation report over an emitted document.

    ``cascade_ir_doc`` is the persisted ``{stem}.cascade_ir.json`` shape (only
    ``region_provenance`` is read); ``html`` is the FINAL emitted accessible HTML.

    **Every verdict is derived from those two PERSISTED artifacts alone.**
    ``emitter_report`` is the optional per-emitter self-report (converted /
    declined / accepted / refused counts); it is CORROBORATION — it sharpens a
    message, never decides a verdict. This is load-bearing: the self-report is
    in-memory BY DESIGN (that is what buys the byte-identical off-path), so it is
    unavailable to ANY out-of-band audit of a persisted corpus — which is the
    natural operator path. A gate whose verdict depended on it would grade the
    same document two different ways depending on how it was invoked, and would
    fire a FABRICATED critical on the out-of-band path (the D1 defect).

    ``document_passed`` is TRI-STATE and the tri-state is the contract:
    ``True`` (every check ran, none critical) / ``False`` (a critical was
    MEASURED) / ``None`` (>= 1 check COULD NOT LOOK — no measurement, never a
    pass, never a green).

    Pure + deterministic. No LLM. No DecisionCapture. No mutation of any input.
    """
    emitter = emitter_report if isinstance(emitter_report, dict) else {}
    regions = [(kind, text) for kind, text in _iter_region_texts(cascade_ir_doc) if text]
    texts = [text for _kind, text in regions]
    # The math evidence is scoped to the text that can actually REACH the render:
    # a declared-drop region (running header / folio) is destroyed by design, so
    # its spans are not conservable (see _RENDER_DROPPED_REGION_KINDS).
    math_texts = [
        text
        for kind, text in regions
        if kind not in _RENDER_DROPPED_REGION_KINDS
    ]
    recall_min = resolve_section_recall_min()

    # An EMPTY region universe is "no measurement", NOT "verified safe" — the
    # conformance-audit skip-count posture. Without an IR there is no evidence
    # side to the conservation invariant, so the two evidence-based checks say so
    # explicitly instead of silently passing (or, worse, reporting every emitted
    # <thead> as an over-emit against a zero denominator).
    if texts:
        math_check = _check_math(math_texts, html, emitter)
        table_check = _check_table_headers(texts, html, emitter)
    else:
        no_ir = {
            "verdict": "not_evaluated",
            "severity": None,
            "message": (
                "no region_provenance in the cascade IR — there is no evidence "
                "universe to conserve against (NO MEASUREMENT, not a pass)"
            ),
        }
        math_check = dict(no_ir)
        table_check = dict(no_ir)

    checks: Dict[str, Any] = {
        "math": math_check,
        "table_headers": table_check,
        "sections": _check_sections(html, recall_min),
        "residue": _check_residue(html, emitter),
        "title_purity": _check_title_purity(html, sibling_title),
        "images": _check_images(cascade_ir_doc, html),
    }

    issues: List[Dict[str, Any]] = []
    not_evaluated: List[str] = []
    for name, check in checks.items():
        severity = check.get("severity")
        # D4 — A GATE THAT CANNOT EVALUATE MUST NEVER CONTRIBUTE A GREEN. This is
        # the founding sin of the whole campaign reproduced one layer up: the
        # pre-T0 `quality.json` self-certified 1.0 over a document nothing had
        # looked at, and `unit_coverage` passed over 352 missing figures. A check
        # that did not run is NO MEASUREMENT, not a pass — so it surfaces as its
        # own LOUD issue severity and it POISONS `document_passed` to null
        # (tri-state), which is neither a pass nor a fabricated failure.
        if check.get("verdict") == "not_evaluated":
            not_evaluated.append(name)
            issues.append(
                {
                    "code": f"AFFORDANCE_{name.upper()}_NOT_EVALUATED",
                    "check": name,
                    "severity": "not_evaluated",
                    "message": check.get("message", ""),
                }
            )
            continue
        if severity not in ("critical", "warning"):
            continue
        issues.append(
            {
                "code": f"AFFORDANCE_{name.upper()}_{str(check.get('verdict')).upper()}",
                "check": name,
                "severity": severity,
                "message": check.get("message", ""),
            }
        )
    critical = [i for i in issues if i["severity"] == "critical"]
    warnings = [i for i in issues if i["severity"] == "warning"]

    # Tri-state, and the tri-state IS the contract:
    #   False -> a critical affordance defect was MEASURED.
    #   None  -> the gate COULD NOT LOOK at >=1 check. NOT a pass. Never green.
    #   True  -> every check ran and none is critical.
    if critical:
        document_passed: Optional[bool] = False
    elif not_evaluated:
        document_passed = None
    else:
        document_passed = True

    return {
        "schema": _SCHEMA,
        "gate": "affordance_conservation",
        "enabled": True,
        "document_passed": document_passed,
        "not_evaluated_checks": not_evaluated,
        "not_evaluated_count": len(not_evaluated),
        "n_regions": len(texts),
        "section_recall_min": recall_min,
        "checks": checks,
        "issues": issues,
        "critical_count": len(critical),
        "warning_count": len(warnings),
        "limits": [
            "audits attrition AFTER the cascade result is serialized",
            "cannot see GENUINE ABSENCE (page-raster figures: evidence ~= 0)",
            "cannot see upstream structural destruction unless the evidence "
            "survives as TEXT (hence the outline-anchored section predicate)",
            "the images check is SHADOW-ONLY and is not an all-clear",
        ],
    }


def run_affordance_gate(
    cascade_ir_doc: Any,
    html: str,
    *,
    emitter_report: Optional[Dict[str, Any]] = None,
    sibling_title: Optional[str] = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> Dict[str, Any]:
    """Compute the report and emit LOUD operator log lines (one per issue)."""
    report = audit_affordances(
        cascade_ir_doc,
        html,
        emitter_report=emitter_report,
        sibling_title=sibling_title,
    )
    _marker = {"critical": "FAIL", "warning": "WARN", "not_evaluated": "NOT-EVALUATED"}
    for issue in report["issues"]:
        marker = _marker.get(str(issue["severity"]), "WARN")
        log(f"[affordance] {marker} {issue['check']}: {issue['message']}")
    if report["document_passed"] is False:
        log(
            "[affordance] DOCUMENT FAIL-CLOSED: "
            f"{report['critical_count']} critical affordance defect(s) — the "
            "document ships source notation or an unnavigable structure"
        )
    elif report["document_passed"] is None:
        # NOT a pass. The whole point of the gate is that "nothing objected" and
        # "nothing looked" must never be the same artifact.
        log(
            "[affordance] DOCUMENT NOT EVALUATED: the gate COULD NOT LOOK at "
            f"{report['not_evaluated_count']} check(s) "
            f"({', '.join(report['not_evaluated_checks'])}) — this is NO "
            "MEASUREMENT, not a pass, and it does NOT certify"
        )
    return report

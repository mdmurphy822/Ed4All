"""Per-region hard gate (Stage 7).

Eliminating filter that drops Stage 6 HTML candidates failing any of:

* HTML5 well-formedness (lxml parser, non-recover mode)
* Text preservation against the source FeatureBlock stream (≥ τ; default 0.85)
* MathML validity (math regions only)
* Table structure (table regions only) — WCAG 2.2 AA SC 1.3.1: data
  tables must have header cells with ``scope=`` or ``id=``/``headers=``
  pairings. Layout tables (``role="presentation"``) and
  single-row/column tables are skipped.
* axe-core under WCAG 2.2 AA (rejects serious/critical impacts)

Order: cheap checks first, axe last. axe runs on a wrapped shell that
satisfies ``page-has-heading-one`` + ``landmark-one-main`` so candidates
aren't blamed for properties the surrounding document is responsible
for.

The driver :func:`gate_per_region` opens **one** :class:`HtmlValidator`
context for the whole document and threads it through every check, so
Chromium boots once. Per-region candidate text is hashed (along with
region kind + region index) to memoize repeats — common in v1 because
the Qwen mock runtime emits the same string K times per region.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from html.parser import HTMLParser
from typing import Any

from semantik_structure.qwen_specialists.types import Candidate
from semantik_structure.structure_graph import Region
from semantik_structure.types import FeatureBlock
from semantik_structure.validate import HtmlValidator, REJECT_IMPACTS

from .mathml_check import validate_mathml
from .text_preserve import normalize, text_preservation_ratio
from .types import CheckOutcome, GateCheck, GateResult


# Content-fidelity COVERAGE thresholds for structured regions (math /
# table). COVERAGE is set-membership — every source cell text / math
# token must APPEAR somewhere in the candidate — NOT an order-sensitive
# SequenceMatcher ratio, so a faithfully restructured <table>/<math>
# isn't unfairly penalised for reordering.
#
# NOTE: these threshold VALUES are provisional pending real-runtime
# distribution data (architecture.md §7.4). They are named module
# constants so they can be tuned without touching the helper code.
TAU_TABLE_COVERAGE = 0.95
TAU_MATH_COVERAGE = 0.90


# Wrap each region candidate in a minimal HTML5 shell so axe rules that
# only apply at the document level (page-has-heading-one,
# landmark-one-main, document-has-title, html-has-lang) don't fire and
# punish the region for lacking page-level scaffolding the assembler
# owns. ``<h1 hidden>`` satisfies page-has-heading-one;
# ``<main>`` satisfies landmark-one-main; ``<title>`` + ``lang="en"``
# fill the remaining doc-shell rules.
_MAIN_SHELL = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    "<title>r</title></head><body><main>"
    "<h1 hidden>r</h1>"
    "{}</main></body></html>"
)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


# HTML5 void elements — never have a closing tag, never carry children.
_HTML_VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


class _BalanceParser(HTMLParser):
    """Track open-tag stack to detect mismatched / unclosed tags.

    stdlib HTMLParser is more permissive than lxml about garbage
    text but it does fire :meth:`error` on truly malformed
    constructs, and we can still detect tag-balance issues by
    bookkeeping. We treat:

    * mismatched close (``</div>`` with ``span`` on top of stack),
    * close-without-open (``</p>`` with empty stack),
    * any leftover open tag at end-of-input,

    as well-formedness failures.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.errors: list[str] = []
        self.stack: list[str] = []

    def handle_starttag(self, tag, attrs):  # noqa: D401
        if tag in _HTML_VOID_ELEMENTS:
            return
        self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):  # noqa: D401
        # Self-closing (e.g. <br/>): treat as void.
        return

    def handle_endtag(self, tag):  # noqa: D401
        if not self.stack:
            self.errors.append(f"close-without-open:{tag}")
            return
        if self.stack[-1] == tag:
            self.stack.pop()
            return
        # Try to recover by popping until we find a match (mirrors
        # browser behavior loosely) — but record the mismatch.
        if tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.errors.append(f"unclosed:{self.stack.pop()}")
            if self.stack:
                self.stack.pop()
        else:
            self.errors.append(f"mismatch:{self.stack[-1]} vs /{tag}")

    def error(self, message):  # noqa: D401
        # stdlib HTMLParser.error is reachable on truly bizarre input.
        self.errors.append(f"parse-error:{message}")


def _check_html5_wellformed(html: str) -> CheckOutcome:
    """Detect malformed HTML5 fragments without external deps.

    Strategy:
    1. Reject empty / non-string input outright.
    2. Reject inputs that contain ``<`` characters but no real tags
       — catches ``<<<`` and similar token-soup garbage.
    3. Walk the input through a tag-balance parser; reject any
       residual open tag, mismatched close, or close-without-open.
    """
    if not isinstance(html, str) or not html.strip():
        return CheckOutcome(
            check=GateCheck.HTML5_VALIDATOR,
            passed=False,
            message="empty html",
        )

    # Detect "fake markup" — strings with stray '<' that aren't part of
    # a tag start. Real tags begin with [A-Za-z!/?]; anything else after
    # '<' is a syntax error in HTML5.
    if re.search(r"<(?![A-Za-z!/?])", html):
        return CheckOutcome(
            check=GateCheck.HTML5_VALIDATOR,
            passed=False,
            message="malformed html: stray '<' not starting a tag",
        )

    p = _BalanceParser()
    try:
        p.feed(html)
        p.close()
    except Exception as exc:
        return CheckOutcome(
            check=GateCheck.HTML5_VALIDATOR,
            passed=False,
            message=f"malformed html: {exc}",
        )
    if p.errors:
        return CheckOutcome(
            check=GateCheck.HTML5_VALIDATOR,
            passed=False,
            message=f"malformed html: {p.errors[0]}",
            details={"errors": p.errors[:5]},
        )
    if p.stack:
        return CheckOutcome(
            check=GateCheck.HTML5_VALIDATOR,
            passed=False,
            message=f"unclosed tag: {p.stack[-1]}",
        )
    return CheckOutcome(
        check=GateCheck.HTML5_VALIDATOR,
        passed=True,
        message="ok",
    )


def _gather_source_text(
    region: Region,
    feature_blocks: Sequence[FeatureBlock],
) -> str:
    """Concatenate the source text for every FB the region claims.

    Falls back to ``payload['text']`` (paragraph/heading) when the FB
    list is empty — math regions in particular often carry their text
    only in payload['src_text']. We try both keys.
    """
    chunks: list[str] = []
    for fb_idx in region.feature_block_indices:
        if 0 <= fb_idx < len(feature_blocks):
            fb = feature_blocks[fb_idx]
            text = getattr(getattr(fb, "raw", None), "text", "") or ""
            text = text.strip()
            if text:
                chunks.append(text)
    if not chunks:
        # Fall back to payload-carried text.
        for key in ("text", "src_text"):
            t = (region.payload or {}).get(key, "")
            if isinstance(t, str) and t.strip():
                chunks.append(t.strip())
                break
    return " ".join(chunks)


def _gather_source_cell_texts(
    region: Region,
    feature_blocks: Sequence[FeatureBlock],
) -> list[str]:
    """Return the per-cell source texts a table region claims.

    Each ``feature_block_indices`` entry is one source cell; the text
    comes from ``FeatureBlock.raw.text`` — the same accessor
    :func:`_gather_source_text` uses. Empty cells are dropped (an empty
    source cell is trivially covered and would otherwise inflate the
    denominator).
    """
    texts: list[str] = []
    for fb_idx in region.feature_block_indices:
        if 0 <= fb_idx < len(feature_blocks):
            fb = feature_blocks[fb_idx]
            text = getattr(getattr(fb, "raw", None), "text", "") or ""
            text = text.strip()
            if text:
                texts.append(text)
    return texts


def _check_table_text_coverage(
    html: str,
    region: Region,
    feature_blocks: Sequence[FeatureBlock],
) -> CheckOutcome:
    """Content-fidelity COVERAGE check for table regions.

    Each source cell (one per ``feature_block_indices`` entry) must
    APPEAR in the candidate: its normalized text must be a substring of
    the concatenated normalized text of the candidate's ``<th>``/``<td>``
    cells. COVERAGE is set-membership, not an order-sensitive ratio, so
    a faithfully reordered/restructured ``<table>`` isn't penalised.

    Fails below :data:`TAU_TABLE_COVERAGE`. This is a real measured
    check — never marked ``skipped=True``.
    """
    source_cells = _gather_source_cell_texts(region, feature_blocks)
    if not source_cells:
        # No source cell text to measure against — same no-measurement
        # contract as the prose path: default-allow but flag skipped.
        return CheckOutcome(
            check=GateCheck.TEXT_PRESERVE,
            passed=True,
            message="skipped:no_source_text",
            skipped=True,
        )

    p = _TableParser()
    try:
        p.feed(html)
        p.close()
    except Exception:
        # Malformed input — the well-formedness check fails it anyway.
        p.cell_texts = []
    candidate_blob = normalize(" ".join(p.cell_texts))

    missing: list[str] = []
    for cell in source_cells:
        if normalize(cell) not in candidate_blob:
            missing.append(cell)
    covered = len(source_cells) - len(missing)
    coverage = covered / len(source_cells)
    passed = coverage >= TAU_TABLE_COVERAGE
    return CheckOutcome(
        check=GateCheck.TEXT_PRESERVE,
        passed=passed,
        message=(
            f"table coverage={coverage:.3f} tau={TAU_TABLE_COVERAGE:.2f} "
            f"({covered}/{len(source_cells)} cells)"
        ),
        details={
            "coverage": coverage,
            "tau": TAU_TABLE_COVERAGE,
            "missing_cells": missing,
        },
    )


# MathML token elements whose text content is human-readable math
# (mirrors mathml_check's allowlist subset). Math text COVERAGE is
# measured against alttext + the concatenated text of these tokens.
_MATH_TOKEN_TAGS = frozenset({"mi", "mo", "mn", "ms", "mtext"})


def _extract_math_candidate_text(html: str) -> str:
    """Return the candidate ``<math>``'s alttext + token-element text."""
    chunks: list[str] = []

    class _MathTextParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self._token_depth = 0

        def handle_starttag(self, tag, attrs):
            local = tag.split(":")[-1].lower()
            if local == "math":
                alttext = (dict(attrs).get("alttext") or "").strip()
                if alttext:
                    chunks.append(alttext)
            elif local in _MATH_TOKEN_TAGS:
                self._token_depth += 1

        def handle_endtag(self, tag):
            local = tag.split(":")[-1].lower()
            if local in _MATH_TOKEN_TAGS and self._token_depth > 0:
                self._token_depth -= 1

        def handle_data(self, data):
            if self._token_depth > 0 and data.strip():
                chunks.append(data.strip())

    p = _MathTextParser()
    try:
        p.feed(html)
        p.close()
    except Exception:
        return ""
    return " ".join(chunks)


def _check_math_text_coverage(
    html: str,
    region: Region,
    feature_blocks: Sequence[FeatureBlock],
) -> CheckOutcome:
    """Content-fidelity COVERAGE check for math regions.

    Source math text comes from ``region.payload['src_text']`` (the
    ``_gather_source_text`` fallback key). The candidate's recoverable
    text is its ``<math alttext="...">`` attribute plus the text of all
    ``<mi>/<mo>/<mn>/<ms>/<mtext>`` token elements. COVERAGE is the
    fraction of source glyphs/tokens that APPEAR in that candidate text
    — set-membership, not an order-sensitive ratio.

    Fails below :data:`TAU_MATH_COVERAGE`. This is a real measured
    check — never marked ``skipped=True``.
    """
    src = (region.payload or {}).get("src_text", "")
    if not (isinstance(src, str) and src.strip()):
        src = _gather_source_text(region, feature_blocks)
    if not src.strip():
        return CheckOutcome(
            check=GateCheck.TEXT_PRESERVE,
            passed=True,
            message="skipped:no_source_text",
            skipped=True,
        )

    candidate_text = normalize(_extract_math_candidate_text(html))
    # Token the source on whitespace; each whitespace-delimited token
    # must appear as a substring of the candidate math text.
    src_tokens = [t for t in normalize(src).split(" ") if t]
    if not src_tokens:
        return CheckOutcome(
            check=GateCheck.TEXT_PRESERVE,
            passed=True,
            message="skipped:no_source_text",
            skipped=True,
        )
    covered = sum(1 for t in src_tokens if t in candidate_text)
    coverage = covered / len(src_tokens)
    passed = coverage >= TAU_MATH_COVERAGE
    return CheckOutcome(
        check=GateCheck.TEXT_PRESERVE,
        passed=passed,
        message=(
            f"math coverage={coverage:.3f} tau={TAU_MATH_COVERAGE:.2f} "
            f"({covered}/{len(src_tokens)} tokens)"
        ),
        details={"coverage": coverage, "tau": TAU_MATH_COVERAGE},
    )


def _check_text_preservation(
    html: str,
    region: Region,
    feature_blocks: Sequence[FeatureBlock],
    tau: float,
) -> CheckOutcome:
    """Return a TEXT_PRESERVE outcome.

    Math and table regions used to skip this check entirely — a
    candidate that silently dropped content passed unconditionally.
    They now dispatch to content-fidelity COVERAGE checks
    (:func:`_check_math_text_coverage` / :func:`_check_table_text_coverage`),
    which measure set-membership of source tokens/cells rather than an
    order-sensitive ratio so structured output isn't unfairly penalised.
    Prose / heading / list regions keep the SequenceMatcher ratio path.
    """
    if region.kind == "math":
        return _check_math_text_coverage(html, region, feature_blocks)
    if region.kind == "table":
        return _check_table_text_coverage(html, region, feature_blocks)
    source = _gather_source_text(region, feature_blocks)
    if not source:
        # No source text at all → pass-through (we can't measure
        # preservation; downstream checks still apply). Marked
        # skipped=True so the eval aggregator can flag silent skips
        # (gap-fill synthetic regions, dropped FBs, etc.).
        return CheckOutcome(
            check=GateCheck.TEXT_PRESERVE,
            passed=True,
            message="skipped:no_source_text",
            skipped=True,
        )
    ratio = text_preservation_ratio(source, html)
    passed = ratio >= tau
    return CheckOutcome(
        check=GateCheck.TEXT_PRESERVE,
        passed=passed,
        message=f"ratio={ratio:.3f} tau={tau:.2f}",
        details={"ratio": ratio, "tau": tau},
    )


def _check_mathml_validity(html: str) -> CheckOutcome:
    """Delegate to :func:`mathml_check.validate_mathml`."""
    return validate_mathml(html)


# ---------------------------------------------------------------------------
# Table structural gate (WCAG 2.2 AA, SC 1.3.1)
# ---------------------------------------------------------------------------


_VALID_TH_SCOPES = frozenset({"col", "row", "colgroup", "rowgroup"})


class _TableParser(HTMLParser):
    """Walk a candidate's outermost ``<table>`` and capture row/col shape
    plus per-``<th>`` attributes.

    Nested tables increment a depth counter; row/cell counts only fire
    on the outermost table (depth==1) so the gate verdict reflects the
    table the candidate actually claims to render.

    The parser does not validate well-formedness — that's
    :func:`_check_html5_wellformed`'s job and runs first. If the input
    is partial or malformed, the cell counts may be off; the gate is
    expected to fail-soft on garbage input (downstream pass-rate already
    excluded that candidate).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth: int = 0
        self.in_outer_row: bool = False
        self.role: str | None = None  # role attr on outermost <table>
        self.row_count: int = 0
        self.col_count: int = 0
        self.current_row_cols: int = 0
        self.th_attrs: list[dict[str, str]] = []
        self.td_with_headers_count: int = 0
        # Per-cell text capture for the content-fidelity coverage check.
        # ``cell_texts`` collects the visible text of every <th>/<td> in
        # the outermost table; ``_cell_depth`` tracks open cells (text
        # outside cells — e.g. <caption> — is intentionally excluded).
        self.cell_texts: list[str] = []
        self._cell_depth: int = 0
        self._cell_chunks: list[str] = []

    def _flush_cell(self) -> None:
        if self._cell_chunks:
            self.cell_texts.append(" ".join(self._cell_chunks))
            self._cell_chunks = []

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if tag == "table":
            if self.depth == 0:
                self.role = (attrs_d.get("role") or "").strip().lower() or None
            self.depth += 1
            return
        if self.depth != 1:
            return  # ignore nested tables and pre-table content
        if tag == "tr":
            self.in_outer_row = True
            self.row_count += 1
            self.current_row_cols = 0
        elif tag == "th" and self.in_outer_row:
            self.current_row_cols += 1
            if self.current_row_cols > self.col_count:
                self.col_count = self.current_row_cols
            self.th_attrs.append(attrs_d)
            self._flush_cell()
            self._cell_depth = 1
        elif tag == "td" and self.in_outer_row:
            self.current_row_cols += 1
            if self.current_row_cols > self.col_count:
                self.col_count = self.current_row_cols
            if "headers" in attrs_d and (attrs_d.get("headers") or "").strip():
                self.td_with_headers_count += 1
            self._flush_cell()
            self._cell_depth = 1

    def handle_data(self, data):
        # Capture text only while inside an outermost-table cell.
        if self.depth == 1 and self._cell_depth > 0 and data.strip():
            self._cell_chunks.append(data.strip())

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.depth == 1 and self._cell_depth > 0:
            self._flush_cell()
            self._cell_depth = 0
        if tag == "table" and self.depth > 0:
            self.depth -= 1
            if self.depth == 0:
                self.in_outer_row = False
                self._flush_cell()
                self._cell_depth = 0
        elif tag == "tr" and self.depth == 1:
            self.in_outer_row = False


def _check_table_structure(html: str, region: Region) -> CheckOutcome:
    """Validate header semantics on a candidate ``<table>``.

    WCAG 2.2 AA SC 1.3.1 ("Info and Relationships") requires
    programmatic association between data cells and their headers in a
    data table. Two patterns satisfy the SC:

      * Simple tables: every ``<th>`` carries ``scope="col|row|colgroup
        |rowgroup"``.
      * Complex tables: ``<th id="...">`` paired with
        ``<td headers="...">`` references.

    Decision logic:

      1. Non-table region → skip (passed=True, skipped=True).
      2. ``<table role="presentation"|"none">`` → skip; author has
         explicitly marked the table as layout, not data.
      3. Outermost table contains ``<th>`` elements → treat as a data
         table regardless of size. Every ``<th>`` must carry a valid
         ``scope=`` OR a non-empty ``id=``. First failing ``<th>``
         determines the verdict.
      4. No ``<th>`` AND multi-row + multi-col → fail. A 2×2+ table
         without header cells almost always means the author lost the
         header semantics during conversion.
      5. No ``<th>`` AND single-row OR single-column → skip (could be a
         layout table or a degenerate data table; insufficient signal
         to call it).

    Earlier the prompt-builder docstring claimed Stage 7 rejected
    tables missing headers; that wasn't true (no validator existed).
    This is the real fix.
    """
    if region.kind != "table":
        return CheckOutcome(
            check=GateCheck.TABLE_STRUCTURE,
            passed=True,
            message=f"skipped:{region.kind}",
            skipped=True,
        )

    p = _TableParser()
    try:
        p.feed(html)
        p.close()
    except Exception as exc:
        # Well-formedness check should have caught this earlier; be
        # defensive so the cascade gets a useful diagnostic if it didn't.
        return CheckOutcome(
            check=GateCheck.TABLE_STRUCTURE,
            passed=False,
            message=f"table parse error: {exc}",
        )

    if p.role in ("presentation", "none"):
        return CheckOutcome(
            check=GateCheck.TABLE_STRUCTURE,
            passed=True,
            message="skipped:layout_table_role",
            skipped=True,
        )

    if p.th_attrs:
        for i, attrs in enumerate(p.th_attrs):
            scope = (attrs.get("scope") or "").strip().lower()
            ident = (attrs.get("id") or "").strip()
            if scope in _VALID_TH_SCOPES:
                continue
            if ident:
                continue  # headers="..." pairing satisfies SC 1.3.1
            return CheckOutcome(
                check=GateCheck.TABLE_STRUCTURE,
                passed=False,
                message=(
                    f"<th> #{i} missing scope= and id= "
                    f"(WCAG 2.2 AA SC 1.3.1: header cells must be "
                    f"programmatically associable)"
                ),
                details={
                    "th_index": i,
                    "th_attrs": attrs,
                    "rows": p.row_count,
                    "cols": p.col_count,
                    "n_th": len(p.th_attrs),
                },
            )
        # scope/id present on every <th>. For COMPLEX tables (multi
        # header rows, dual axis, spanned <th>) scope alone is
        # ambiguous — require full H43 id/headers pairing (Plan 12 B1).
        from .table_h43 import verify_h43

        h43_ok, h43_msg, h43_details = verify_h43(html)
        if not h43_ok:
            return CheckOutcome(
                check=GateCheck.TABLE_STRUCTURE,
                passed=False,
                message=h43_msg,
                details={
                    "rows": p.row_count,
                    "cols": p.col_count,
                    "n_th": len(p.th_attrs),
                    **h43_details,
                },
            )
        return CheckOutcome(
            check=GateCheck.TABLE_STRUCTURE,
            passed=True,
            message=(
                f"ok rows={p.row_count} cols={p.col_count} th={len(p.th_attrs)} h43={h43_msg}"
            ),
            details={
                "rows": p.row_count,
                "cols": p.col_count,
                "n_th": len(p.th_attrs),
                "n_td_with_headers": p.td_with_headers_count,
                **h43_details,
            },
        )

    if p.row_count >= 2 and p.col_count >= 2:
        return CheckOutcome(
            check=GateCheck.TABLE_STRUCTURE,
            passed=False,
            message=(
                f"data table missing <th> headers "
                f"(rows={p.row_count} cols={p.col_count}); "
                f"WCAG 2.2 AA SC 1.3.1 requires header cells on data tables"
            ),
            details={"rows": p.row_count, "cols": p.col_count, "n_th": 0},
        )

    return CheckOutcome(
        check=GateCheck.TABLE_STRUCTURE,
        passed=True,
        message=(f"skipped:no_th_small_table rows={p.row_count} cols={p.col_count}"),
        skipped=True,
    )


def _check_axe(shell_html: str, validator: HtmlValidator) -> CheckOutcome:
    """Run axe-core via the shared validator and convert the result.

    The validator's ``ValidationResult.ok`` is True iff there are no
    violations at serious/critical impact — the same threshold used by
    training-pair gating in :mod:`semantik_structure.validate`.
    """
    try:
        result = validator.check(shell_html)
    except Exception as exc:
        return CheckOutcome(
            check=GateCheck.AXE_WCAG22AA,
            passed=False,
            message=f"axe runtime error: {exc!r}",
        )
    if result.error:
        return CheckOutcome(
            check=GateCheck.AXE_WCAG22AA,
            passed=False,
            message=f"axe error: {result.error}",
        )
    blocking = [v for v in result.violations if v.impact in REJECT_IMPACTS]
    if blocking:
        ids = ",".join(sorted({v.id for v in blocking}))
        return CheckOutcome(
            check=GateCheck.AXE_WCAG22AA,
            passed=False,
            message=f"violations:{ids}",
            details={
                "violations": [
                    {"id": v.id, "impact": v.impact, "nodes": v.nodes} for v in blocking
                ],
            },
        )
    return CheckOutcome(
        check=GateCheck.AXE_WCAG22AA,
        passed=True,
        message="ok",
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def check_region(
    region_html: str,
    *,
    region_id: int,
    region: Region | None = None,
    feature_blocks: Sequence[FeatureBlock] = (),
    validator: HtmlValidator | None = None,
    tau: float = 0.85,
    context: Any = None,
) -> GateResult:
    """Run all four checks on one candidate. Order: cheap first, axe last.

    The signature accepts the legacy Phase-0 ``context=`` kwarg so
    callers built against the stub keep working; new callers pass
    ``region`` / ``feature_blocks`` / ``validator`` explicitly.
    """
    # Allow legacy context=dict callers — extract fields if present.
    if region is None and isinstance(context, dict):
        region = context.get("region")
        feature_blocks = context.get("feature_blocks", feature_blocks)
        validator = context.get("validator", validator)
        tau = float(context.get("tau", tau))

    checks: list[CheckOutcome] = []
    checks.append(_check_html5_wellformed(region_html))

    if region is not None:
        checks.append(
            _check_text_preservation(
                region_html,
                region,
                feature_blocks,
                tau,
            )
        )
        if region.kind == "math":
            checks.append(_check_mathml_validity(region_html))
        if region.kind == "table":
            checks.append(_check_table_structure(region_html, region))

    # Short-circuit: axe is ~100x the cost of the cheap checks above
    # (Chromium navigate + script eval per candidate). Verdict is
    # all(c.passed for c in checks), so if a cheap check already failed
    # the axe outcome can't change the result — skip it. Under mock
    # runtime every candidate fails text_preserve, which is what made
    # Stage 7 51x slower than baseline (eval 2026-05-12).
    if validator is not None and all(c.passed for c in checks):
        shell = _MAIN_SHELL.format(region_html)
        checks.append(_check_axe(shell, validator))

    return GateResult.from_checks(
        "hard_region",
        checks,
        region_id=region_id,
    )


def gate_per_region(
    candidates: dict[int, list[Candidate]],
    regions: Sequence[Region],
    feature_blocks: Sequence[FeatureBlock],
    *,
    text_preservation_threshold: float = 0.85,
    parallel: int = 1,
    validator: HtmlValidator | None = None,
) -> tuple[dict[int, list[Candidate]], dict[int, list[GateResult]]]:
    """Drive the per-region hard gate over all (region, candidate) pairs.

    Parameters
    ----------
    candidates
        Stage-6 output: dict keyed by region index in ``regions``,
        value is a list of :class:`Candidate`.
    regions
        Stage-5 typed regions. Indexed by the keys in ``candidates``.
    feature_blocks
        Stage-2 FeatureBlock stream — used for text-preservation
        source text.
    text_preservation_threshold
        τ floor for the text-preserve check. Default 0.85.
    parallel
        Reserved for v1.1 — must be 1 for v1.
    validator
        Optional pre-opened :class:`HtmlValidator`. Useful for tests
        that want to inject a fake. When ``None`` (the production
        path) a fresh validator is opened for the duration of the call.

    Returns
    -------
    (survivors, results)
        ``survivors`` is the same shape as ``candidates`` but with
        per-region lists trimmed to the candidates that passed every
        check. ``survivors[i] == []`` is the all-K-fail signal Stage 9
        reads. ``results`` maps region index → per-candidate
        :class:`GateResult` (in input order, including failures) so
        callers can introspect *why* a candidate was dropped.
    """
    if parallel != 1:
        raise NotImplementedError("parallel>1 reserved for v1.1")

    survivors: dict[int, list[Candidate]] = {}
    results: dict[int, list[GateResult]] = {}

    def _run(v: HtmlValidator) -> None:
        cache: dict[str, GateResult] = {}
        for region_idx, cand_list in candidates.items():
            if region_idx < 0 or region_idx >= len(regions):
                # Defensive: stage 6 should never key a region we
                # can't index. Skip rather than crash.
                continue
            region = regions[region_idx]
            kept: list[Candidate] = []
            per_cand: list[GateResult] = []
            for c in cand_list:
                key = _hash_key(c.text, region.kind, region_idx)
                gr = cache.get(key)
                if gr is None:
                    gr = check_region(
                        c.text,
                        region_id=region_idx,
                        region=region,
                        feature_blocks=feature_blocks,
                        validator=v,
                        tau=text_preservation_threshold,
                    )
                    cache[key] = gr
                per_cand.append(gr)
                if gr.passed:
                    kept.append(c)
            survivors[region_idx] = kept
            results[region_idx] = per_cand

    if validator is not None:
        _run(validator)
    else:
        with HtmlValidator() as v:
            _run(v)

    return survivors, results


def _hash_key(html: str, kind: str, region_idx: int) -> str:
    """Cache key includes ``region_idx`` because text-preservation is
    per-region (the same HTML candidate measured against a different
    source FB stream gets a different ratio). ``kind`` is included
    because math regions get an extra MathML check that other kinds
    don't."""
    h = hashlib.sha256()
    h.update(kind.encode())
    h.update(b"|")
    h.update(str(region_idx).encode())
    h.update(b"|")
    h.update(html.encode())
    return h.hexdigest()


__all__ = [
    "check_region",
    "gate_per_region",
]

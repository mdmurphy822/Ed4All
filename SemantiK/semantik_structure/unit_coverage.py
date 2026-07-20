"""SEMANTIK_UNIT_COVERAGE_GATE — deterministic content-loss fidelity gate (task #43).

theta (the Stage-12 DeBERTa cross-encoder) is the historic semantic-preservation
guardrail, but it is stub-collapsed (the v8 mode-collapse flat-0.7 placeholder)
so it carries no trustworthy signal and is OUT of the trust chain. This gate is
its deterministic,
model-free replacement for the ONE thing theta was supposed to catch:
**content loss** — text that entered structure detection (the fused per-page
extraction block/unit texts) but vanished from the emitted HTML (the observed
failure class: a worked example present in extraction, absent from output).

Mechanism (pure, CPU-only, no LLM): after final assembly, tokenize the
extraction-side text universe PER PAGE and the emitted HTML text, then compute
normalized-token coverage per page (share of a page's extraction tokens that
survive anywhere in the output). Pages below :func:`resolve_unit_coverage_min`
emit a loud gate issue naming the LONGEST missing spans; the document
FAIL-CLOSES when any page drops below the harder
:func:`resolve_unit_coverage_page_floor`.

Contract:

* **Gate ``SEMANTIK_UNIT_COVERAGE_GATE`` (default OFF).** Off → the gate never
  runs and nothing is added to the result (byte-identical). It is the operator's
  opt-in fidelity guardrail; theta stays wired in code but is superseded as the
  content-loss signal (a ``theta bypassed`` note is logged when the gate runs).
* **Pure / deterministic → NO ``DecisionCapture``** (flagged loudly here).
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable, Iterable

__all__ = [
    "SEMANTIK_UNIT_COVERAGE_GATE_ENV",
    "SEMANTIK_UNIT_COVERAGE_MIN_ENV",
    "SEMANTIK_UNIT_COVERAGE_PAGE_FLOOR_ENV",
    "resolve_unit_coverage_gate_mode",
    "resolve_unit_coverage_min",
    "resolve_unit_coverage_page_floor",
    "compute_unit_coverage",
    "run_unit_coverage_gate",
]

SEMANTIK_UNIT_COVERAGE_GATE_ENV = "SEMANTIK_UNIT_COVERAGE_GATE"
SEMANTIK_UNIT_COVERAGE_MIN_ENV = "SEMANTIK_UNIT_COVERAGE_MIN"
SEMANTIK_UNIT_COVERAGE_PAGE_FLOOR_ENV = "SEMANTIK_UNIT_COVERAGE_PAGE_FLOOR"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_DEFAULT_MIN = 0.90
_DEFAULT_PAGE_FLOOR = 0.70

# Cap on the number of missing spans reported per page (longest first) and the
# character prefix kept per span — a bounded, operator-readable issue.
_MAX_SPANS_PER_PAGE = 10
_SPAN_PREFIX_CHARS = 80

# Alphanumeric (unicode) token — drops punctuation / whitespace / markup.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
# Word-ish original substrings (for reconstructing a readable missing span).
_WORD_SPLIT_RE = re.compile(r"\S+")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def resolve_unit_coverage_gate_mode() -> bool:
    """Whether the unit-coverage fidelity gate may run (default OFF).

    Parse-with-fallback truthy set; unset / blank / falsey / garbage → off
    (byte-identical — the gate never runs). Read at CALL time.
    """
    return (os.environ.get(SEMANTIK_UNIT_COVERAGE_GATE_ENV) or "").strip().lower() in _TRUTHY


def _resolve_float(env: str, default: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    raw = os.environ.get(env)
    if raw is None or not str(raw).strip():
        return default
    try:
        val = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if val != val or val in (float("inf"), float("-inf")):  # NaN / Inf
        return default
    if val < lo or val > hi:
        return default
    return val


def resolve_unit_coverage_min() -> float:
    """Per-page coverage floor below which a page emits a WARNING gate issue
    (default ``0.90``). Float parse-with-fallback in ``[0, 1]``."""
    return _resolve_float(SEMANTIK_UNIT_COVERAGE_MIN_ENV, _DEFAULT_MIN)


def resolve_unit_coverage_page_floor() -> float:
    """Hard per-page coverage floor below which the DOCUMENT fails closed
    (default ``0.70``). Float parse-with-fallback in ``[0, 1]``."""
    return _resolve_float(SEMANTIK_UNIT_COVERAGE_PAGE_FLOOR_ENV, _DEFAULT_PAGE_FLOOR)


def _normalize_tokens(text: str) -> list[str]:
    """Lowercased alphanumeric-unicode tokens (markup/punctuation stripped)."""
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def strip_html_text(html: str) -> str:
    """Best-effort plain text from assembled HTML (tags removed, entities
    unescaped, whitespace collapsed)."""
    if not html:
        return ""
    import html as _html_mod

    txt = _TAG_RE.sub(" ", html)
    txt = _html_mod.unescape(txt)
    return _WS_RE.sub(" ", txt).strip()


def _missing_spans(block_text: str, emitted: set[str]) -> list[str]:
    """Maximal runs of consecutive words in ``block_text`` whose normalized
    token is NOT in the emitted token set, reconstructed as readable spans."""
    spans: list[str] = []
    run: list[str] = []
    for word in _WORD_SPLIT_RE.findall(block_text or ""):
        toks = _normalize_tokens(word)
        # A word with no alphanumeric content (pure punctuation) never breaks or
        # extends a run — it is neither missing nor covered.
        if not toks:
            continue
        if all(t not in emitted for t in toks):
            run.append(word)
        else:
            if run:
                spans.append(" ".join(run))
                run = []
    if run:
        spans.append(" ".join(run))
    return spans


def compute_unit_coverage(
    extraction_units: Iterable[tuple[int, str]],
    emitted_html: str,
    *,
    min_coverage: float | None = None,
    page_floor: float | None = None,
) -> dict[str, Any]:
    """Deterministic per-page content-loss report.

    ``extraction_units`` is ``(page, text)`` for every extraction block/unit that
    entered structure detection (synthetic image blocks excluded by the caller).
    ``emitted_html`` is the final assembled HTML. Returns the gate report dict
    (see the module docstring / ``run_unit_coverage_gate``).
    """
    if min_coverage is None:
        min_coverage = resolve_unit_coverage_min()
    if page_floor is None:
        page_floor = resolve_unit_coverage_page_floor()

    emitted_set = set(_normalize_tokens(strip_html_text(emitted_html)))

    # Group extraction texts by page (preserving order for span reconstruction).
    by_page: dict[int, list[str]] = {}
    for page, text in extraction_units:
        if text is None:
            continue
        by_page.setdefault(int(page), []).append(str(text))

    pages_report: list[dict[str, Any]] = []
    warned_pages: list[int] = []
    failed_pages: list[int] = []

    for page in sorted(by_page):
        block_texts = by_page[page]
        page_tokens: list[str] = []
        for bt in block_texts:
            page_tokens.extend(_normalize_tokens(bt))
        n_tokens = len(page_tokens)
        if n_tokens == 0:
            # Nothing to lose on this page (image-only / empty) → full coverage.
            pages_report.append(
                {
                    "page": page,
                    "coverage": 1.0,
                    "n_tokens": 0,
                    "n_missing": 0,
                    "below_min": False,
                    "below_floor": False,
                    "missing_spans": [],
                }
            )
            continue
        n_missing = sum(1 for t in page_tokens if t not in emitted_set)
        coverage = (n_tokens - n_missing) / n_tokens
        below_min = coverage < min_coverage
        below_floor = coverage < page_floor

        missing_spans: list[str] = []
        if below_min:
            raw_spans: list[str] = []
            for bt in block_texts:
                raw_spans.extend(_missing_spans(bt, emitted_set))
            # Longest (by char length) first; bounded + prefix-truncated.
            raw_spans.sort(key=len, reverse=True)
            for span in raw_spans[:_MAX_SPANS_PER_PAGE]:
                missing_spans.append(span[:_SPAN_PREFIX_CHARS])

        pages_report.append(
            {
                "page": page,
                "coverage": round(coverage, 4),
                "n_tokens": n_tokens,
                "n_missing": n_missing,
                "below_min": below_min,
                "below_floor": below_floor,
                "missing_spans": missing_spans,
            }
        )
        if below_floor:
            failed_pages.append(page)
        elif below_min:
            warned_pages.append(page)

    document_passed = not failed_pages
    return {
        "gate": "unit_coverage",
        "enabled": True,
        "min_coverage": min_coverage,
        "page_floor": page_floor,
        "n_pages": len(pages_report),
        "document_passed": document_passed,
        "warned_pages": warned_pages,
        "failed_pages": failed_pages,
        "pages": pages_report,
    }


def run_unit_coverage_gate(
    extraction_units: Iterable[tuple[int, str]],
    emitted_html: str,
    *,
    log: Callable[[str], None] = lambda _msg: None,
) -> dict[str, Any]:
    """Compute the coverage report and emit LOUD operator log lines.

    Logs one ``theta bypassed`` note (this gate supersedes the stub-collapsed
    theta as the content-loss signal), a per-below-min-page
    line naming the longest missing span, and a document-level FAIL-CLOSED line
    when any page dropped below the hard floor.
    """
    log("[unit-coverage] theta bypassed: unit-coverage gate active")
    report = compute_unit_coverage(extraction_units, emitted_html)
    for pr in report["pages"]:
        if pr["below_min"]:
            top = pr["missing_spans"][0] if pr["missing_spans"] else ""
            severity = "FAIL" if pr["below_floor"] else "WARN"
            log(
                f"[unit-coverage] {severity} page {pr['page']} coverage="
                f"{pr['coverage']:.3f} (<{report['min_coverage']}); "
                f"{pr['n_missing']} tokens missing; longest missing span: "
                f"{top!r}"
            )
    if not report["document_passed"]:
        log(
            "[unit-coverage] DOCUMENT FAIL-CLOSED: pages "
            f"{report['failed_pages']} below hard floor {report['page_floor']}"
        )
    return report

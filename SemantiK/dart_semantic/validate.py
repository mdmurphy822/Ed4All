"""WCAG 2.2 AA validation gate — the quality floor for training data.

Any candidate training pair must have its HTML form pass axe-core with zero
critical-or-serious violations under the wcag22aa ruleset. Pairs that fail are
dropped, not repaired. Configured conservatively: we accept 'minor' and
'moderate' findings (mostly advisory) but reject 'serious' and 'critical'.

Usage:
    from dart_semantic.validate import validate_html, ValidationResult

    result = validate_html(html_string)
    if result.ok:
        save_training_pair(...)
    else:
        log.drop(result.reasons)

A reusable HtmlValidator can batch many documents through a single Chromium
context (much faster when gating 10K+ pairs).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from axe_playwright_python.sync_playwright import Axe
from playwright.sync_api import sync_playwright

# WCAG 2.0/2.1/2.2 tags — explicit set so the ruleset doesn't drift if
# axe-core's defaults change. Include 'best-practice' only for logging
# (it isn't a conformance requirement).
ACCEPT_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]

# Impacts that cause a pair to be rejected from the training set.
REJECT_IMPACTS = {"serious", "critical"}


@dataclass
class Violation:
    id: str
    impact: str
    help: str
    nodes: int

    def __str__(self) -> str:
        return f"{self.impact}:{self.id} ({self.nodes} nodes) — {self.help}"


@dataclass
class ValidationResult:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    error: str | None = None  # runtime error, if any

    @property
    def reasons(self) -> list[str]:
        if self.error:
            return [f"runtime error: {self.error}"]
        return [str(v) for v in self.violations if v.impact in REJECT_IMPACTS]


def _result_from_axe(raw: dict) -> ValidationResult:
    """Convert axe-core's raw JSON report into our decision record."""
    violations = []
    for v in raw.get("violations", []):
        violations.append(Violation(
            id=v.get("id", ""),
            impact=v.get("impact") or "unknown",
            help=v.get("help", ""),
            nodes=len(v.get("nodes", [])),
        ))
    blocking = [v for v in violations if v.impact in REJECT_IMPACTS]
    return ValidationResult(ok=not blocking, violations=violations)


class HtmlValidator:
    """Reusable validator that keeps a Chromium browser + context open.

    Use as a context manager when validating many docs in a row:

        with HtmlValidator() as v:
            for doc in candidates:
                result = v.check(emit_html.emit(doc))
                ...

    A single-shot call is available via the module-level `validate_html()`
    helper — it spins up and tears down a browser per call (fine for a handful
    of checks, ~100x slower than batch mode over thousands).
    """

    def __init__(self, accept_tags: list[str] = None):
        self._tags = accept_tags or ACCEPT_TAGS
        self._pw = None
        self._browser = None
        self._context = None
        self._axe = Axe()

    def __enter__(self) -> "HtmlValidator":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        self._context = self._browser.new_context()
        return self

    def __exit__(self, *exc) -> None:
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def check(self, html: str) -> ValidationResult:
        page = self._context.new_page()
        try:
            page.set_content(html, wait_until="load")
            # axe-playwright-python json.dumps()s options itself — pass the dict.
            options = {
                "runOnly": {"type": "tag", "values": self._tags},
                "resultTypes": ["violations"],
            }
            results = self._axe.run(page, options=options)
            # AxeResults wraps the response dict in .response
            raw_dict = results.response if hasattr(results, "response") else results
            return _result_from_axe(raw_dict)
        except Exception as exc:
            return ValidationResult(ok=False, error=repr(exc))
        finally:
            page.close()

    def render_pdf(self, html: str, out_path) -> None:
        """Render HTML to PDF using the same Chromium instance the validator holds.

        Provided here because Python's sync Playwright API can't be nested — a
        caller already inside a `with HtmlValidator()` block cannot open a
        second sync_playwright context. Reuse this one.
        """
        page = self._context.new_page()
        try:
            page.set_content(html, wait_until="networkidle")
            page.pdf(path=str(out_path), format="Letter", print_background=True)
        finally:
            page.close()


def validate_html(html: str, *, accept_tags: list[str] = None) -> ValidationResult:
    """One-shot validation. For >1 document, use HtmlValidator as a context manager."""
    with HtmlValidator(accept_tags=accept_tags) as v:
        return v.check(html)


@contextmanager
def validator() -> Iterator[HtmlValidator]:
    """Convenience wrapper: `with validator() as v: v.check(html)`."""
    with HtmlValidator() as v:
        yield v

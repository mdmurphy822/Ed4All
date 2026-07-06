"""Document-level HARD gate (Stage 10).

Eliminating filter on the assembled document from Stage 9. Runs six
binary checks; any failure drops the document. The driver
:func:`gate_document` is the public entry point.

Checks (in cheap-first order):

1. ``HTML5_VALIDATOR``  — well-formed HTML5; ``<!DOCTYPE html>`` present.
2. ``LANG_DECLARED``    — ``<html lang="...">`` non-empty.
3. ``TITLE_PRESENT``    — ``<title>...</title>`` non-empty.
4. ``LANDMARK_PRESENCE``— at least one ``<main>``.
5. ``HEADING_TREE``     — first heading is ``<h1>``; no level skips; in 1..6.
6. ``NO_EXTERNAL_REFS`` — no external ``http(s)://`` resource ``src=`` (img /
   script) or stylesheet ``<link href>``, and no residual Markdown image.
7. ``AXE_WCAG22AA``     — axe-core full doc, no serious/critical violations.

Stage 10 receives a *full document* (not a fragment), so we do **not**
wrap candidates in the ``_MAIN_SHELL`` used by Stage 7 — the doc owns
its own shell. axe is run last because it boots Chromium and is by far
the most expensive check.

A region_id of ``None`` is the doc-level sentinel (matches
:class:`GateResult` defaults).
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

from dart_semantic.assembler.types import AssembledDoc
from dart_semantic.validate import HtmlValidator, REJECT_IMPACTS

from .types import CheckOutcome, GateCheck, GateResult


# ---------------------------------------------------------------------------
# HTML5 well-formedness — duplicated from hard_region (NOT shared) so
# Stage 7 and Stage 10 stay independent. Keeping a private copy avoids
# a refactor that would touch Stage 7 tests, per the task constraints.
# ---------------------------------------------------------------------------


_HTML_VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})


_DOCTYPE_RE = re.compile(r"<!doctype\s+html\b[^>]*>", re.IGNORECASE)


class _BalanceParser(HTMLParser):
    """Track open-tag stack to detect mismatched / unclosed tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.errors: list[str] = []
        self.stack: list[str] = []

    def handle_starttag(self, tag, attrs):  # noqa: D401
        if tag in _HTML_VOID_ELEMENTS:
            return
        self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):  # noqa: D401
        return

    def handle_endtag(self, tag):  # noqa: D401
        if not self.stack:
            self.errors.append(f"close-without-open:{tag}")
            return
        if self.stack[-1] == tag:
            self.stack.pop()
            return
        if tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.errors.append(f"unclosed:{self.stack.pop()}")
            if self.stack:
                self.stack.pop()
        else:
            self.errors.append(f"mismatch:{self.stack[-1]} vs /{tag}")

    def error(self, message):  # noqa: D401
        self.errors.append(f"parse-error:{message}")


def _check_html5_wellformed(html: str) -> CheckOutcome:
    """Doc-level well-formedness: requires non-empty input,
    ``<!DOCTYPE html>``, no stray ``<``, and balanced tags."""
    if not isinstance(html, str) or not html.strip():
        return CheckOutcome(
            check=GateCheck.HTML5_VALIDATOR,
            passed=False,
            message="empty html",
        )

    if not _DOCTYPE_RE.search(html):
        return CheckOutcome(
            check=GateCheck.HTML5_VALIDATOR,
            passed=False,
            message="missing <!DOCTYPE html>",
        )

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


# ---------------------------------------------------------------------------
# <html lang="...">
# ---------------------------------------------------------------------------


# Match <html ... lang="value" ...> or single quoted; case-insensitive on tag
# name and attribute key. We require a non-empty value.
_HTML_TAG_RE = re.compile(r"<html\b([^>]*)>", re.IGNORECASE)
_LANG_ATTR_RE = re.compile(
    r"""\blang\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""",
    re.IGNORECASE,
)


def _check_html_lang(html: str) -> CheckOutcome:
    m = _HTML_TAG_RE.search(html)
    if not m:
        return CheckOutcome(
            check=GateCheck.LANG_DECLARED,
            passed=False,
            message="no <html> element",
        )
    attrs = m.group(1) or ""
    lang_m = _LANG_ATTR_RE.search(attrs)
    if not lang_m:
        return CheckOutcome(
            check=GateCheck.LANG_DECLARED,
            passed=False,
            message="<html> missing lang attribute",
        )
    value = (lang_m.group(1) or lang_m.group(2) or lang_m.group(3) or "").strip()
    if not value:
        return CheckOutcome(
            check=GateCheck.LANG_DECLARED,
            passed=False,
            message="<html lang=''> empty",
        )
    return CheckOutcome(
        check=GateCheck.LANG_DECLARED,
        passed=True,
        message=f"lang={value!r}",
        details={"lang": value},
    )


# ---------------------------------------------------------------------------
# <title>
# ---------------------------------------------------------------------------


_TITLE_RE = re.compile(
    r"<title\b[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL,
)


def _check_title_present(html: str) -> CheckOutcome:
    m = _TITLE_RE.search(html)
    if not m:
        return CheckOutcome(
            check=GateCheck.TITLE_PRESENT,
            passed=False,
            message="no <title> element",
        )
    text = (m.group(1) or "").strip()
    if not text:
        return CheckOutcome(
            check=GateCheck.TITLE_PRESENT,
            passed=False,
            message="<title> empty",
        )
    return CheckOutcome(
        check=GateCheck.TITLE_PRESENT,
        passed=True,
        message=f"title length={len(text)}",
        details={"title": text[:120]},
    )


# ---------------------------------------------------------------------------
# Landmarks — at least one <main>.
# ---------------------------------------------------------------------------


_MAIN_OPEN_RE = re.compile(r"<main\b[^>]*>", re.IGNORECASE)
_MAIN_ID_MAIN_CONTENT_RE = re.compile(
    r"""<main\b[^>]*\bid\s*=\s*(?:"main-content"|'main-content'|main-content(?=[\s>]))""",
    re.IGNORECASE,
)


def _check_main_landmark(html: str) -> CheckOutcome:
    matches = _MAIN_OPEN_RE.findall(html)
    if not matches:
        return CheckOutcome(
            check=GateCheck.LANDMARK_PRESENCE,
            passed=False,
            message="no <main> landmark",
        )
    bonus = bool(_MAIN_ID_MAIN_CONTENT_RE.search(html))
    return CheckOutcome(
        check=GateCheck.LANDMARK_PRESENCE,
        passed=True,
        message=f"main_count={len(matches)} skip_link_target={bonus}",
        details={"main_count": len(matches), "main_id_main_content": bonus},
    )


# ---------------------------------------------------------------------------
# Heading hierarchy — h1..h6 in document order, no skips, first is h1.
# ---------------------------------------------------------------------------


_HEADING_OPEN_RE = re.compile(r"<h([1-6])\b[^>]*>", re.IGNORECASE)


def _check_heading_hierarchy(html: str) -> CheckOutcome:
    levels = [int(m.group(1)) for m in _HEADING_OPEN_RE.finditer(html)]
    if not levels:
        # Zero headings is OK at this gate. Stage 9's missing-title
        # detector covers the "no title" case separately. Marked
        # skipped=True because the hierarchy validator never ran —
        # this is a default-allow path, not a measured pass.
        return CheckOutcome(
            check=GateCheck.HEADING_TREE,
            passed=True,
            message="no headings",
            details={"levels": []},
            skipped=True,
        )

    if any(lvl < 1 or lvl > 6 for lvl in levels):
        # Defensive — the regex caps to [1-6] but keep the contract.
        return CheckOutcome(
            check=GateCheck.HEADING_TREE,
            passed=False,
            message="heading level out of 1..6",
            details={"levels": levels},
        )

    if levels[0] != 1:
        return CheckOutcome(
            check=GateCheck.HEADING_TREE,
            passed=False,
            message=f"first heading is h{levels[0]} (expected h1)",
            details={"levels": levels},
        )

    # No skips: each level may stay equal, go up by 1, or drop to any
    # ancestor level. Skipping down (e.g. h1 -> h3) is forbidden.
    prev = levels[0]
    for lvl in levels[1:]:
        if lvl > prev + 1:
            return CheckOutcome(
                check=GateCheck.HEADING_TREE,
                passed=False,
                message=f"level skip h{prev} -> h{lvl}",
                details={"levels": levels},
            )
        prev = lvl

    return CheckOutcome(
        check=GateCheck.HEADING_TREE,
        passed=True,
        message=f"levels ok (n={len(levels)})",
        details={"levels": levels},
    )


# ---------------------------------------------------------------------------
# No external network references — no off-origin resource fetch, no residual
# fabricated Markdown image.
# ---------------------------------------------------------------------------


# An external resource-loading ``src=`` on an <img> / <script> (a fetched
# resource, WCAG 1.1.1 content-loss risk + a CSP off-origin fetch). Prose
# ``<a href>`` links are NOT gated — OpenStax / cnx attribution anchors are
# legitimate content, not a resource fetch.
_EXTERNAL_SRC_RE = re.compile(
    r"""<(?:img|script)\b[^>]*\bsrc\s*=\s*["']?\s*https?://""",
    re.IGNORECASE,
)
# An external stylesheet ``<link ... href="https://...">`` (loads off-origin CSS).
_EXTERNAL_LINK_HREF_RE = re.compile(
    r"""<link\b[^>]*\bhref\s*=\s*["']?\s*https?://""",
    re.IGNORECASE,
)
# A residual Markdown image ``![alt](target)`` (the VLM-fabricated-image escape
# that the adapter sanitizer should have removed).
_RESIDUAL_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _check_no_external_refs(html: str) -> CheckOutcome:
    """Fail a document that fetches an off-origin resource or carries a
    residual Markdown image.

    Guards against the scan-lane fabricated-image defect leaking through to the
    shipped page: an external ``src=`` (``<img>`` / ``<script>``) or stylesheet
    ``<link href>`` is a broken off-origin fetch (WCAG 1.1.1 + CSP), and a
    residual ``![alt](url)`` is an un-sanitized fabricated image. Prose
    ``<a href>`` attribution links are legitimate content and are NOT gated.
    """
    md = _RESIDUAL_MD_IMAGE_RE.search(html)
    if md:
        return CheckOutcome(
            check=GateCheck.NO_EXTERNAL_REFS,
            passed=False,
            message="residual markdown image",
            details={"match": md.group(0)[:120]},
        )
    src = _EXTERNAL_SRC_RE.search(html)
    if src:
        return CheckOutcome(
            check=GateCheck.NO_EXTERNAL_REFS,
            passed=False,
            message="external resource src",
            details={"match": src.group(0)[:120]},
        )
    link = _EXTERNAL_LINK_HREF_RE.search(html)
    if link:
        return CheckOutcome(
            check=GateCheck.NO_EXTERNAL_REFS,
            passed=False,
            message="external stylesheet link href",
            details={"match": link.group(0)[:120]},
        )
    return CheckOutcome(
        check=GateCheck.NO_EXTERNAL_REFS,
        passed=True,
        message="ok",
    )


# ---------------------------------------------------------------------------
# axe-core (full document)
# ---------------------------------------------------------------------------


def _check_axe_full_doc(html: str, validator: HtmlValidator) -> CheckOutcome:
    try:
        result = validator.check(html)
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
                    {"id": v.id, "impact": v.impact, "nodes": v.nodes}
                    for v in blocking
                ],
            },
        )
    return CheckOutcome(
        check=GateCheck.AXE_WCAG22AA,
        passed=True,
        message="ok",
    )


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def gate_document(
    doc: AssembledDoc,
    *,
    validator: HtmlValidator | None = None,
) -> GateResult:
    """Run all six doc-level checks against ``doc.html``.

    Order: cheap (regex / parser) checks first, axe last. axe boots
    Chromium; passing a pre-opened ``validator`` reuses the existing
    browser. When ``validator is None`` a fresh :class:`HtmlValidator`
    is opened for the duration of the call.

    The result's ``passed`` is the AND of every individual check.
    Doc-level gates use ``region_id=None`` (the :class:`GateResult`
    sentinel for "this is not a per-region verdict").
    """
    html = doc.html
    checks: list[CheckOutcome] = []
    checks.append(_check_html5_wellformed(html))
    checks.append(_check_html_lang(html))
    checks.append(_check_title_present(html))
    checks.append(_check_main_landmark(html))
    checks.append(_check_heading_hierarchy(html))
    checks.append(_check_no_external_refs(html))

    if validator is not None:
        checks.append(_check_axe_full_doc(html, validator))
    else:
        with HtmlValidator() as v:
            checks.append(_check_axe_full_doc(html, v))

    return GateResult.from_checks(
        "hard_document", checks, region_id=None,
    )


# Phase-0 stub kept for backward compatibility — anything still calling
# the old ``check_document`` shape now flows through ``gate_document``.
def check_document(doc_html: str, *, assembled: AssembledDoc | None = None) -> GateResult:
    """Legacy alias. Prefer :func:`gate_document`.

    Builds a stand-in :class:`AssembledDoc` (only ``html`` populated)
    when no explicit ``assembled`` is provided — sufficient because
    Stage 10 only ever inspects ``doc.html``.
    """
    if assembled is None:
        assembled = AssembledDoc(
            html=doc_html,
            gaps_found=[],
            gaps_resolved=[],
            gaps_fallback=[],
            heading_tree=[],
            landmarks={},
            anchors={},
            region_provenance=[],
            sub_task_log={},
        )
    return gate_document(assembled)


__all__ = [
    "check_document",
    "gate_document",
]

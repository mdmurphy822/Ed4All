"""Source-page viewer service — serve an archived citation page safely.

This is the serving wrapper around
``lib.retrieval.citation_anchor.resolve_source_page``. It exists because the
underlying resolver is a *read-only library* that takes an ``item_path`` and
walks the course dir / cartridge zips with **no traversal guard**
(``_resolve_dart_page`` does ``root / item_path`` + ``.is_file()`` directly).
A learner-facing route hands ``item_path`` straight from a URL query string,
so this wrapper closes the hole *before* the resolver touches the filesystem
(pre-rejection) and *after* it returns an on-disk path (commonpath
containment), mirroring the uploads-router precedent
(``gui/routers/uploads.py::_validate_within_uploads`` +
``delete_upload`` dot-dot pre-rejection).

Archived cartridge HTML is **untrusted**: it can carry ``<script>``, inline
event handlers, ``javascript:`` URLs, and external loaders. The wrapper strips
all of those (bs4) and returns restrictive CSP / nosniff headers so the route
layer can echo them. It also injects heading ``id``s using the **same** slug
algorithm as ``grounded_answer._fragment_for`` so a ``#fragment`` from a
citation link lands on the cited section.

Frozen public contract (Executor A codes the route against this):

    render_source_page(slug, item_path, libv2_root, fragment=None)
        -> SourcePageResult(html: str, media_type: str, headers: dict)

    raises SourcePageError(status: int, code: str, detail: str)

No network, no LLM, no decision capture (deterministic, read-only path).
Heavy imports (bs4, the anchor resolver) are deferred to call time so the
module imports clean on a default install (retrieval_service precedent).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# --------------------------------------------------------------------------- #
# Public result / error types (frozen surface)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourcePageResult:
    """The viewer-wrapped archived page + the headers the route must return."""

    html: str
    media_type: str = "text/html; charset=utf-8"
    headers: Dict[str, str] = field(default_factory=dict)


class SourcePageError(Exception):
    """Mapped to an HTTP error by the route. ``code`` is the flat-wire error
    code; ``detail`` is the operator-actionable message."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Slug shape for a course (matches the GUI/route precedent: lower-first, then
# the usual slug alphabet). A learner can't reach an arbitrary dir name.
_SLUG_RE = re.compile(r"^[a-z0-9][a-zA-Z0-9._-]*$")

# Windows drive-letter prefix (e.g. ``C:\\`` or ``C:/``) — treated as absolute.
_DRIVE_RE = re.compile(r"^[A-Za-z]:")

# Restrictive Content-Security-Policy for the archived (untrusted) page. No
# scripts at all; images only same-origin or data: URIs; inline styles allowed
# (archived pages carry presentational style attributes we don't strip).
_CSP = "default-src 'none'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"

_RESPONSE_HEADERS: Dict[str, str] = {
    "Content-Security-Policy": _CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

# Elements stripped wholesale from untrusted cartridge HTML.
_STRIP_TAGS = ("script", "iframe", "object", "embed", "noscript", "base")

# URL-bearing attributes we scrub for ``javascript:`` (and other active) schemes.
_URL_ATTRS = ("href", "src", "action", "formaction", "xlink:href")

# Active/dangerous URL schemes rejected in URL-bearing attributes.
_ACTIVE_SCHEME_RE = re.compile(r"^\s*(?:javascript|vbscript|data):", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Slug helper (pinned equal to grounded_answer._fragment_for's algorithm)
# --------------------------------------------------------------------------- #


def heading_slug(heading: str) -> str:
    """Slug a heading exactly as ``grounded_answer._fragment_for`` does.

    ``re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")``. Pinned by a
    unit test against ``_fragment_for`` output so fragment navigation lands on
    the cited section.
    """
    return re.sub(r"[^a-z0-9]+", "-", str(heading).lower()).strip("-")


# --------------------------------------------------------------------------- #
# Path safety (closes the resolver's traversal hole)
# --------------------------------------------------------------------------- #


def _validate_slug(slug: str, libv2_root: Path) -> Path:
    """Validate the course slug shape + existence; return the resolved course dir.

    Raises ``SourcePageError`` (422 for a malformed slug, 404 for an unknown
    course). The resolved course dir is commonpath-checked under ``courses/``
    so a slug that survived the regex but escaped (it can't, but defence in
    depth) is rejected.
    """
    if not slug or not _SLUG_RE.match(slug):
        raise SourcePageError(422, "invalid_item_path", f"invalid course slug: {slug!r}")
    courses_dir = (Path(libv2_root) / "courses").resolve()
    course_dir = (courses_dir / slug).resolve()
    try:
        common = os.path.commonpath([str(course_dir), str(courses_dir)])
    except ValueError:
        raise SourcePageError(422, "invalid_item_path", f"slug escapes courses root: {slug!r}") from None
    if common != str(courses_dir):
        raise SourcePageError(422, "invalid_item_path", f"slug escapes courses root: {slug!r}")
    if not course_dir.is_dir():
        raise SourcePageError(404, "course_not_found", f"course not found: {slug}")
    return course_dir


def _reject_traversal(item_path: str) -> str:
    """Pre-reject a hostile ``item_path`` BEFORE any fs/zip access.

    Rejects (raising 422 ``invalid_item_path``): empty, NUL bytes, backslashes,
    absolute paths (POSIX root or Windows drive letter), and any ``..`` path
    segment. Mirrors ``uploads.delete_upload``'s dot-dot pre-rejection — the
    point is to never *reach* the unguarded resolver with an escaping path.
    """
    if not item_path:
        raise SourcePageError(422, "invalid_item_path", "item_path is required")
    if "\x00" in item_path:
        raise SourcePageError(422, "invalid_item_path", "item_path contains a NUL byte")
    if "\\" in item_path:
        raise SourcePageError(422, "invalid_item_path", "item_path contains a backslash")
    if item_path.startswith("/") or _DRIVE_RE.match(item_path):
        raise SourcePageError(422, "invalid_item_path", f"absolute item_path rejected: {item_path!r}")
    # ``Path.is_absolute`` catches platform-specific absolute forms too.
    if Path(item_path).is_absolute():
        raise SourcePageError(422, "invalid_item_path", f"absolute item_path rejected: {item_path!r}")
    # Any ``..`` *segment* (a bare ``..``, or ``a/../b``) — split on forward
    # slash since backslashes were already rejected.
    segments = item_path.split("/")
    if any(seg == ".." for seg in segments):
        raise SourcePageError(422, "invalid_item_path", f"item_path escapes course dir: {item_path!r}")
    return item_path


def _assert_contained(resolved: Path, course_dir: Path) -> None:
    """Post-resolution commonpath containment for on-disk hits.

    The resolver's basename ``rglob`` fallback (and any symlink under a source
    root) can surface a path outside the course dir even when ``item_path``
    looked clean. Re-resolve and commonpath-check — this catches symlink
    escapes the pre-rejection can't see.
    """
    real = resolved.resolve()
    base = course_dir.resolve()
    try:
        common = os.path.commonpath([str(real), str(base)])
    except ValueError:  # different drives / mixed roots
        raise SourcePageError(
            422, "invalid_item_path", f"resolved page escapes course dir: {real}"
        ) from None
    if common != str(base):
        raise SourcePageError(
            422, "invalid_item_path", f"resolved page escapes course dir: {real}"
        )


# --------------------------------------------------------------------------- #
# Sanitize + wrap (untrusted archived HTML → viewer shell)
# --------------------------------------------------------------------------- #


def _sanitize_and_wrap(
    raw_html: str, *, slug: str, page_label: str, fragment: Optional[str]
) -> str:
    """Strip active content, inject heading ids, prepend the banner nav.

    bs4-based. Drops ``<script>``/``<iframe>``/``<object>``/``<embed>``/
    ``<noscript>``/``<base>`` elements, every ``on*`` event-handler attribute,
    and ``javascript:``/``vbscript:``/``data:`` URLs in URL-bearing attributes.
    Injects ``id``s on headings (only where missing) using the
    ``_fragment_for`` slug. Prepends a heading-free ``<nav>`` banner so the
    archived page's own ``<h1>`` stays the unique document h1.
    """
    from bs4 import BeautifulSoup  # noqa: PLC0415

    soup = BeautifulSoup(raw_html, "html.parser")

    # 1) Drop dangerous elements entirely.
    for tag_name in _STRIP_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()

    # 2) Scrub attributes on every remaining element.
    for el in soup.find_all(True):
        attrs = dict(el.attrs)
        for name, value in attrs.items():
            lname = name.lower()
            # Inline event handlers (onclick, onmouseover, ...).
            if lname.startswith("on"):
                del el[name]
                continue
            # Active URL schemes in URL-bearing attributes.
            if lname in _URL_ATTRS:
                val = value if isinstance(value, str) else " ".join(value)
                if _ACTIVE_SCHEME_RE.match(val):
                    del el[name]
            # link[rel=preload]/[rel=prefetch]/[rel=import] active loaders.
            if el.name == "link":
                rel = el.get("rel")
                rel_val = " ".join(rel) if isinstance(rel, (list, tuple)) else str(rel or "")
                if rel_val.lower() in ("preload", "prefetch", "import", "modulepreload"):
                    el.decompose()
                    break

    # 3) Inject heading ids (only where missing) using the frozen slug.
    for level in ("h1", "h2", "h3", "h4", "h5", "h6"):
        for heading in soup.find_all(level):
            if heading.get("id"):
                continue
            text = heading.get_text(" ", strip=True)
            slug_id = heading_slug(text)
            if slug_id:
                heading["id"] = slug_id

    # 4) Ensure <html lang> present (mark when injected so the gate can tell).
    html_el = soup.find("html")
    if html_el is not None and not html_el.get("lang"):
        html_el["lang"] = "en"
        html_el["data-lang-injected"] = "true"

    # 5) Build the banner nav (no heading element — preserves the single h1).
    banner = _banner_markup(slug=slug, page_label=page_label, fragment=fragment)
    body = soup.find("body")
    if body is not None:
        banner_soup = BeautifulSoup(banner, "html.parser")
        body.insert(0, banner_soup)
    else:
        # No <body> — wrap the whole thing.
        return banner + str(soup)

    return str(soup)


def _banner_markup(*, slug: str, page_label: str, fragment: Optional[str]) -> str:
    """The viewer banner: a heading-free landmark nav.

    Carries the course slug, page label, a fragment note when the citation
    pointed at a non-heading (xpath/None → top of page), and a "Back to your
    question" link. All dynamic strings escaped.
    """
    from html import escape  # noqa: PLC0415

    note = ""
    if fragment is None:
        # No heading fragment — the cited location is somewhere on this page.
        note = '<p class="viewer-note">The cited section is on this page.</p>'

    return (
        '<nav class="source-viewer-banner" aria-label="Source context">'
        f'<p class="viewer-course">Course: {escape(slug)}</p>'
        f'<p class="viewer-page">Page: {escape(page_label)}</p>'
        f"{note}"
        '<p class="viewer-back">'
        '<a href="/learn/">Back to your question</a>'
        "</p>"
        "</nav>"
    )


# --------------------------------------------------------------------------- #
# Public entry point (frozen)
# --------------------------------------------------------------------------- #


def render_source_page(
    slug: str,
    item_path: str,
    libv2_root: Path,
    fragment: Optional[str] = None,
) -> SourcePageResult:
    """Serve an archived citation source page, viewer-wrapped and sanitized.

    1. Validate the slug shape + course existence (422 / 404).
    2. Pre-reject a hostile ``item_path`` before any fs/zip access (422).
    3. Resolve via the WS1 anchor resolver (imscc members read in-memory).
    4. Commonpath-contain on-disk hits (catches symlink escapes) (422).
    5. Sanitize + wrap + inject heading ids; return content + CSP headers.

    Raises ``SourcePageError(status, code, detail)``.
    """
    libv2_root = Path(libv2_root)
    course_dir = _validate_slug(slug, libv2_root)
    item_path = _reject_traversal(item_path)

    # Lazy import: keeps this module import-clean on a default install and
    # wraps (never edits) the WS1 read-only resolver.
    from lib.retrieval.citation_anchor import resolve_source_page  # noqa: PLC0415
    from lib.retrieval.grounded_answer import (  # noqa: PLC0415
        _infer_chunkset_kind,
        page_label_for_source,
    )

    chunkset_kind = _infer_chunkset_kind(libv2_root, slug)
    chunk = {"source": {"item_path": item_path}}

    try:
        resolved = resolve_source_page(chunk, course_dir, chunkset_kind=chunkset_kind)
    except SourcePageError:
        raise
    except Exception as exc:  # resolver blew up on a malformed page/zip
        raise SourcePageError(500, "source_render_failed", f"resolution error: {exc}") from exc

    if resolved is None:
        raise SourcePageError(
            404, "source_page_missing", f"no archived page for item_path: {item_path}"
        )

    source_id, raw_html = resolved

    # On-disk hits (DART / pre-unpacked) → commonpath containment. Zip-member
    # hits are inherently contained (in-memory read; member name came from
    # ``namelist()``) and are not filesystem paths.
    if isinstance(source_id, Path):
        _assert_contained(source_id, course_dir)

    page_label = page_label_for_source({"item_path": item_path})

    try:
        wrapped = _sanitize_and_wrap(
            raw_html, slug=slug, page_label=page_label, fragment=fragment
        )
    except SourcePageError:
        raise
    except Exception as exc:
        raise SourcePageError(500, "source_render_failed", f"sanitization error: {exc}") from exc

    return SourcePageResult(
        html=wrapped,
        media_type="text/html; charset=utf-8",
        headers=dict(_RESPONSE_HEADERS),
    )


__all__ = [
    "SourcePageResult",
    "SourcePageError",
    "render_source_page",
    "heading_slug",
]

"""Source-materials service — inventory + serving of original archived sources.

The Studio course generated from a textbook is a *transformed derivative*; this
service surfaces the **original** archived sources behind it: the DART-converted
accessible HTML (block-anchored so a citation can deep-link to the exact source
position) and the raw PDFs. It backs three routes on ``gui.routers.library``:

* ``GET /api/courses/{id}/source-materials`` → the per-course inventory.
* ``GET /api/courses/{id}/source-doc?doc=&ref=`` → one sanitized DART document,
  with serve-time block-anchor injection (``id="dart-{block_id}"``) + figure-src
  rewrite so a ``#dart-<block>`` fragment lands on the cited block.
* ``GET /api/courses/{id}/source-doc-asset?doc=&path=`` → a figure asset from the
  doc's ``{stem}_figures/`` dir.

It is additive and read-only. Everything security-sensitive — the slug shape,
traversal pre-rejection, commonpath containment, the active-content scrub, the
asset-extension whitelist — is **reused** from the audited precedents
(``source_page``, ``imscc_service``, ``source_pdf``) rather than re-implemented.

Path-safety contract (copied from the precedents): the course slug is
regex-validated + commonpath-contained under ``courses/``; ``doc`` is whitelisted
against the inventory stem listing (NEVER used as a path); ``path`` (figure
assets) is pre-rejected (no ``..``/absolute/NUL/backslash) before any fs read and
commonpath-contained under the figures dir. ``manifest.json::source_artifacts``
paths are absolute machine paths and are **never** read — discovery re-derives
from dir listings (the ``source_pdf`` precedent).

Operator toggle (``ED4ALL_SOURCE_MATERIALS``, default on; per-course
``manifest.json::viewer.expose_source_materials`` override): surfacing the
verbatim textbook is a redistribution act the generated course is not. When
disabled, the inventory returns ``enabled:false`` + empty lists and the doc /
asset endpoints 403.

No network, no LLM, no decision capture (deterministic, read-only). Heavy
imports (bs4) are deferred to call time so the module imports clean on a default
install.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the audited sanitiser + heading-id slug + the typed error.
from gui.services.source_page import (
    SourcePageError,
    _RESPONSE_HEADERS,
    heading_slug,
    sanitize_soup,
)

# Reuse the asset-extension whitelist + asset CSP from the IMSCC viewer.
from gui.services.imscc_service import _ASSET_CONTENT_TYPES, _ASSET_HEADERS

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Course-slug shape (identical to source_page / imscc_service / source_pdf).
_SLUG_RE = re.compile(r"^[a-z0-9][a-zA-Z0-9._-]*$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")

# Archived DART HTML search roots (first root wins per stem). Pinned equal to
# ``lib.retrieval.citation_anchor._DART_ROOTS`` — re-declared as a literal so
# this module stays import-light (no retrieval-stack dependency at import time);
# a unit test asserts the two stay in sync.
_DART_ROOTS = ("sources/textbooks", "source/html", "source/dart")

# Toggle env var (default ON). Per-course override beats env.
_TOGGLE_ENV = "ED4ALL_SOURCE_MATERIALS"

# Service error type — reuse SourcePageError so the router maps one
# (status, code, detail) contract across every viewer path.
SourceMaterialsError = SourcePageError


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ServedDoc:
    """A served source document or figure asset + headers the route echoes back."""

    body: bytes
    media_type: str
    headers: Dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Root / slug resolution (path-safe; mirrors the precedents)
# --------------------------------------------------------------------------- #


def _libv2_root() -> Path:
    """Resolve the LibV2 root, honoring ``ED4ALL_LIBV2_ROOT`` (call-time read)."""
    override = os.environ.get("ED4ALL_LIBV2_ROOT")
    if override:
        return Path(override)
    from lib.paths import LIBV2_PATH  # noqa: PLC0415

    return Path(LIBV2_PATH)


def _validate_slug(slug: str, libv2_root: Path) -> Path:
    """Validate slug shape + course existence; return the resolved course dir.

    Raises ``SourceMaterialsError`` (422 malformed, 404 unknown). Commonpath-
    checks the resolved course dir under ``courses/`` (defence in depth),
    mirroring ``source_page._validate_slug``.
    """
    if not slug or not _SLUG_RE.match(slug):
        raise SourceMaterialsError(422, "invalid_course_id", f"invalid course slug: {slug!r}")
    courses_dir = (Path(libv2_root) / "courses").resolve()
    course_dir = (courses_dir / slug).resolve()
    try:
        common = os.path.commonpath([str(course_dir), str(courses_dir)])
    except ValueError:
        raise SourceMaterialsError(
            422, "invalid_course_id", f"slug escapes courses root: {slug!r}"
        ) from None
    if common != str(courses_dir):
        raise SourceMaterialsError(
            422, "invalid_course_id", f"slug escapes courses root: {slug!r}"
        )
    if not course_dir.is_dir():
        raise SourceMaterialsError(404, "course_not_found", f"course not found: {slug}")
    return course_dir


# --------------------------------------------------------------------------- #
# Toggle resolution (§2.5)
# --------------------------------------------------------------------------- #


def _env_enabled() -> bool:
    """Read the ``ED4ALL_SOURCE_MATERIALS`` env flag (default ON).

    Truthy: unset / ``on`` / ``1`` / ``true`` / ``yes`` (case-insensitive). Any
    explicit ``off`` / ``0`` / ``false`` / ``no`` disables. Garbage falls back to
    enabled (the documented default-on posture).
    """
    raw = os.environ.get(_TOGGLE_ENV)
    if raw is None:
        return True
    val = raw.strip().lower()
    if val in ("off", "0", "false", "no"):
        return False
    return True


def _course_override(course_dir: Path) -> Optional[bool]:
    """Per-course ``manifest.json::viewer.expose_source_materials`` override.

    Returns ``True``/``False`` when the key is an explicit bool, else ``None``
    (absent / malformed manifest → inherit the env flag). Best-effort: a missing
    or unreadable manifest never raises (the inventory must still serve).
    """
    manifest = course_dir / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    viewer = data.get("viewer")
    if not isinstance(viewer, dict):
        return None
    val = viewer.get("expose_source_materials")
    if isinstance(val, bool):
        return val
    return None


def is_enabled(course_dir: Path) -> bool:
    """Resolve the effective toggle: per-course override beats the env flag."""
    override = _course_override(course_dir)
    if override is not None:
        return override
    return _env_enabled()


# --------------------------------------------------------------------------- #
# DART-doc + PDF discovery (dir listings — never source_artifacts paths)
# --------------------------------------------------------------------------- #


def _iter_dart_roots(course_dir: Path):
    """Yield existing DART roots in ``_DART_ROOTS`` order (first root wins)."""
    for rel in _DART_ROOTS:
        cand = course_dir / rel
        if cand.is_dir():
            yield rel, cand


def _doc_title(html_path: Path) -> str:
    """Cheap ``<title>`` parse from a bounded head read; humanized-stem fallback.

    Reads only the first ~8 KB (titles live in ``<head>``) and regex-matches the
    title to avoid parsing a 300 KB document just for a label. Falls back to the
    humanized filename stem (``foo-bar_accessible`` → ``Foo Bar Accessible``).
    """
    try:
        with html_path.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(8192)
    except OSError:
        head = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", head, re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        if title:
            return title
    stem = html_path.stem
    humanized = re.sub(r"[-_]+", " ", stem).strip()
    return humanized.title() if humanized else stem


def _discover_dart_docs(course_dir: Path) -> List[dict]:
    """Discover DART docs across the search roots (first root wins per stem).

    Each entry: ``{doc, title, root, has_figures, size}`` where ``doc`` is the
    filename stem (the whitelist key the ``?doc=`` query is matched against,
    NEVER a path). ``has_figures`` is the presence of a sibling ``{stem}_figures/``
    dir (the Wave-19 figure layout). Sorted by ``doc`` for stable output.
    """
    seen: Dict[str, dict] = {}
    for rel, root in _iter_dart_roots(course_dir):
        for html_path in sorted(root.glob("*.html")):
            if not html_path.is_file():
                continue
            stem = html_path.stem
            if stem in seen:  # first root wins
                continue
            figures = root / f"{stem}_figures"
            try:
                size = html_path.stat().st_size
            except OSError:
                size = 0
            seen[stem] = {
                "doc": stem,
                "title": _doc_title(html_path),
                "root": rel,
                "has_figures": figures.is_dir(),
                "size": size,
            }
    return [seen[k] for k in sorted(seen)]


def _resolve_doc_path(course_dir: Path, doc: str) -> Tuple[Path, Path]:
    """Whitelist ``doc`` against the discovered stem listing → ``(html, root_dir)``.

    ``doc`` is matched ONLY against the inventory stems (never used as a path),
    then the on-disk hit is commonpath-contained under the course dir. Raises
    ``SourceMaterialsError`` (422 a hostile/malformed ``doc``, 404 an unknown
    stem).
    """
    if not doc or not doc.strip():
        raise SourceMaterialsError(422, "invalid_doc", "doc is required")
    # Defence in depth — we only ever match a stem from the listing, but a
    # path-shaped doc must never resolve.
    if "/" in doc or "\\" in doc or ".." in doc or "\x00" in doc:
        raise SourceMaterialsError(422, "invalid_doc", f"invalid source doc: {doc!r}")
    for rel, root in _iter_dart_roots(course_dir):
        cand = root / f"{doc}.html"
        if cand.is_file():
            _assert_contained(cand, course_dir)
            return cand, root
    raise SourceMaterialsError(404, "source_doc_missing", f"no archived source doc: {doc!r}")


def _assert_contained(resolved: Path, course_dir: Path) -> None:
    """Post-resolution commonpath containment (catches symlink escapes).

    Mirrors ``source_page._assert_contained`` — a symlink under a source root
    could surface a path outside the course dir even when the stem looked clean.
    """
    real = resolved.resolve()
    base = course_dir.resolve()
    try:
        common = os.path.commonpath([str(real), str(base)])
    except ValueError:
        raise SourceMaterialsError(
            422, "invalid_doc", f"resolved doc escapes course dir: {real}"
        ) from None
    if common != str(base):
        raise SourceMaterialsError(
            422, "invalid_doc", f"resolved doc escapes course dir: {real}"
        )


# --------------------------------------------------------------------------- #
# Inventory (plan item 1)
# --------------------------------------------------------------------------- #


def inventory(course_dir: Path) -> dict:
    """Per-course source-materials inventory (pure, read-only).

    Returns ``{dart_docs: [...], pdfs: [...], provenance: bool}``. Discovers DART
    docs by walking ``_DART_ROOTS`` (first root wins per stem) and PDFs via the
    reused ``source_pdf`` listing helpers — never the absolute/stale
    ``manifest.json::source_artifacts`` paths. An empty inventory is a valid
    result (chunks-only / provenance-without-artifacts archives). ``provenance``
    reports ``manifest.json::features.source_provenance`` (advisory).
    """
    from gui.services import source_pdf as _pdf  # noqa: PLC0415

    dart_docs = _discover_dart_docs(course_dir)

    pdfs: List[dict] = []
    pdf_dir = _pdf._pdf_dir(course_dir)
    if pdf_dir is not None:
        for p in _pdf._list_pdfs(pdf_dir):
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            pdfs.append({"name": p.name, "size": size})

    return {
        "dart_docs": dart_docs,
        "pdfs": pdfs,
        "provenance": _provenance_flag(course_dir),
    }


def _provenance_flag(course_dir: Path) -> bool:
    """Advisory ``features.source_provenance`` flag from the manifest (best-effort)."""
    manifest = course_dir / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    features = data.get("features")
    if not isinstance(features, dict):
        return False
    return bool(features.get("source_provenance"))


def list_source_materials(slug: str, libv2_root: Optional[Path] = None) -> dict:
    """Public inventory entry point for the ``/source-materials`` route.

    Returns ``{slug, enabled, dart_docs, pdfs}``. When the operator toggle is off
    (env or per-course override), ``enabled`` is ``False`` and both lists are
    empty (the inventory is suppressed, not just hidden). Raises
    ``SourceMaterialsError`` (422 invalid slug, 404 unknown course).
    """
    root = Path(libv2_root) if libv2_root is not None else _libv2_root()
    course_dir = _validate_slug(slug, root)
    if not is_enabled(course_dir):
        return {"slug": slug, "enabled": False, "dart_docs": [], "pdfs": []}
    inv = inventory(course_dir)
    return {
        "slug": slug,
        "enabled": True,
        "dart_docs": inv["dart_docs"],
        "pdfs": inv["pdfs"],
    }


# --------------------------------------------------------------------------- #
# Doc serving — sanitize + heading-id + block-anchor + figure-src rewrite
# --------------------------------------------------------------------------- #


def _normalize_ref(ref: str) -> str:
    """Slug a ``?ref=`` value for the ``sec-`` anchor form (heading-id slug)."""
    return heading_slug(ref)


def _serve_time_transform(raw_html: str, *, slug: str, doc: str, ref: Optional[str]) -> str:
    """Sanitize + inject heading/block ids + rewrite figure src (the new path).

    Builds on the shared ``sanitize_soup`` scrub + heading-id injection (the
    ``imscc_service._sanitize_page`` shape) with two new serve-time passes:

    * *block-anchor injection*: every element carrying ``data-dart-block-id`` and
      no ``id`` gets ``id="dart-{block_id}"`` so a ``#dart-<block>`` fragment
      lands on the cited block. Existing ids (``id="sec-…"`` etc.) pass through
      untouched (never clobbered).
    * *figure-src rewrite*: a relative ``src="{stem}_figures/…"`` is rewritten to
      the ``/source-doc-asset`` endpoint (a bare relative URL would otherwise
      resolve against the API path and 404). Absolute / data: srcs pass through.

    ``ref`` is informational here — anchor resolution is purely client-side via
    the server-built ``#fragment``; the three-form match (block-id / id /
    ``sec-``+normalized) is what the caller appends to the URL.
    """
    from bs4 import BeautifulSoup  # noqa: PLC0415
    from urllib.parse import quote  # noqa: PLC0415

    soup = BeautifulSoup(raw_html, "html.parser")

    # 1) Shared audited active-content scrub.
    sanitize_soup(soup)

    # 2) Heading ids for in-page anchor navigation. The Ask answer path builds a
    #    fragment by slugging the chunk's ``section_heading`` (the
    #    ``grounded_answer._fragment_for`` heading-slug); that text-slug must
    #    resolve to the heading even when the heading ALREADY carries a different
    #    id (DART stamps hash ids like ``sec-<hash>`` on ~all headings, which the
    #    text-slug fragment would otherwise never match — the "lands at document
    #    TOP" defect). So: a heading with no id gets the text-slug id directly; a
    #    heading whose existing id differs from the text-slug gets the text-slug
    #    planted as an additional sibling anchor span (never clobbering the
    #    original id — page-anchor / footnote ids stay intact).
    for level in ("h1", "h2", "h3", "h4", "h5", "h6"):
        for heading in soup.find_all(level):
            text = heading.get_text(" ", strip=True)
            slug_id = heading_slug(text)
            if not slug_id:
                continue
            existing = heading.get("id")
            if not existing:
                heading["id"] = slug_id
            elif existing != slug_id and not soup.find(id=slug_id):
                anchor = soup.new_tag("span", id=slug_id)
                anchor["class"] = ["dart-heading-anchor"]
                heading.insert(0, anchor)

    # 3) Block-anchor injection: every data-dart-block-id element must end up
    #    reachable via the #dart-{block_id} fragment (the citation deep-link
    #    contract). Free element → set the id; element with its OWN id
    #    (sec-…, fn-… footnote anchors) → never clobber, plant an empty
    #    inline anchor span as its first child instead.
    for el in soup.find_all(attrs={"data-dart-block-id": True}):
        block_id = el.get("data-dart-block-id")
        if isinstance(block_id, (list, tuple)):
            block_id = " ".join(block_id)
        block_id = str(block_id or "").strip()
        if not block_id:
            continue
        dart_id = f"dart-{block_id}"
        if not el.get("id"):
            el["id"] = dart_id
        elif el.get("id") != dart_id:
            anchor = soup.new_tag("span", id=dart_id)
            anchor["class"] = ["dart-block-anchor"]
            el.insert(0, anchor)

    # 3b) Per-page anchor injection: the FIRST element carrying a given
    #     ``data-dart-pages`` value gets ``id="page-{N}"`` so a ``#page-N``
    #     fragment (built by the LO-map / "View in textbook" deep links from the
    #     block's physical page) scrolls there. ``data-dart-pages`` carries the
    #     "3" / "3-5" / "3,5,7" forms; a range/list anchors page-N on the FIRST
    #     element whose attribute STARTS at N (so #page-3 lands on the "3-5"
    #     block). Never clobbers an existing id — plant an inline anchor span as
    #     the first child instead (mirrors the block-anchor contract above).
    #     Page-less docs are byte-identical to today (no anchors injected).
    try:
        from Trainforge.chunker.helpers import parse_dart_pages_attr  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — degrade gracefully if chunker absent
        parse_dart_pages_attr = None  # type: ignore[assignment]
    if parse_dart_pages_attr is not None:
        seen_pages: set = set()
        for el in soup.find_all(attrs={"data-dart-pages": True}):
            attr = el.get("data-dart-pages")
            if isinstance(attr, (list, tuple)):
                attr = " ".join(attr)
            el_pages = parse_dart_pages_attr(attr)
            for pg in el_pages:
                if pg in seen_pages:
                    continue
                seen_pages.add(pg)
                page_id = "page-{}".format(pg)
                if not el.get("id"):
                    el["id"] = page_id
                elif el.get("id") != page_id:
                    anchor = soup.new_tag("span", id=page_id)
                    anchor["class"] = ["dart-page-anchor"]
                    el.insert(0, anchor)

    # 4) Figure-src rewrite: relative {stem}_figures/… → /source-doc-asset.
    figures_prefix = f"{doc}_figures/"
    slug_q = quote(str(slug), safe="")
    doc_q = quote(str(doc), safe="")
    for img in soup.find_all("img"):
        src = img.get("src")
        if not isinstance(src, str) or not src:
            continue
        rel = None
        if src.startswith(figures_prefix):
            rel = src[len(figures_prefix):]
        elif src.startswith("./" + figures_prefix):
            rel = src[len("./" + figures_prefix):]
        if rel is None:
            continue
        rel_q = quote(rel, safe="/")
        img["src"] = (
            f"/api/courses/{slug_q}/source-doc-asset?doc={doc_q}&path={rel_q}"
        )

    # 5) Ensure <html lang> present (mark when injected so the gate can tell).
    html_el = soup.find("html")
    if html_el is not None and not html_el.get("lang"):
        html_el["lang"] = "en"
        html_el["data-lang-injected"] = "true"

    return str(soup)


def serve_source_doc(
    slug: str,
    doc: str,
    ref: Optional[str] = None,
    *,
    libv2_root: Optional[Path] = None,
) -> ServedDoc:
    """Serve one sanitized, block-anchored DART document for the viewer/citation.

    1. Validate slug + course existence (422 / 404).
    2. Toggle gate: 403 ``source_materials_disabled`` when disabled.
    3. Whitelist ``doc`` against the inventory stem listing (422 hostile, 404
       unknown), commonpath-contain the on-disk hit.
    4. Sanitize + inject heading ids + inject ``id="dart-{block}"`` block anchors
       + rewrite figure src; return the HTML with the ``source_page`` CSP headers.

    Raises ``SourceMaterialsError(status, code, detail)``.
    """
    root = Path(libv2_root) if libv2_root is not None else _libv2_root()
    course_dir = _validate_slug(slug, root)
    if not is_enabled(course_dir):
        raise SourceMaterialsError(
            403, "source_materials_disabled", "source materials are disabled for this deployment"
        )
    html_path, _root_dir = _resolve_doc_path(course_dir, doc)

    try:
        raw = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SourceMaterialsError(500, "source_doc_read_error", str(exc)) from exc

    try:
        transformed = _serve_time_transform(raw, slug=slug, doc=doc, ref=ref)
    except SourceMaterialsError:
        raise
    except Exception as exc:  # noqa: BLE001 — a malformed doc / bs4 blow-up
        raise SourceMaterialsError(500, "source_doc_render_failed", str(exc)) from exc

    return ServedDoc(
        body=transformed.encode("utf-8"),
        media_type="text/html; charset=utf-8",
        headers=dict(_RESPONSE_HEADERS),
    )


# --------------------------------------------------------------------------- #
# Figure-asset serving — from the doc's {stem}_figures/ dir only
# --------------------------------------------------------------------------- #


def _reject_asset_path(path: str) -> str:
    """Pre-reject a hostile figure path BEFORE any fs read (mirrors imscc_service)."""
    if not path or not path.strip():
        raise SourceMaterialsError(422, "invalid_asset_path", "path is required")
    if "\x00" in path:
        raise SourceMaterialsError(422, "invalid_asset_path", "path contains a NUL byte")
    if "\\" in path:
        raise SourceMaterialsError(422, "invalid_asset_path", "path contains a backslash")
    if path.startswith("/") or _DRIVE_RE.match(path):
        raise SourceMaterialsError(422, "invalid_asset_path", f"absolute path rejected: {path!r}")
    if Path(path).is_absolute():
        raise SourceMaterialsError(422, "invalid_asset_path", f"absolute path rejected: {path!r}")
    if any(seg == ".." for seg in path.split("/")):
        raise SourceMaterialsError(422, "invalid_asset_path", f"path escapes figures dir: {path!r}")
    return path


def serve_source_doc_asset(
    slug: str,
    doc: str,
    path: str,
    *,
    libv2_root: Optional[Path] = None,
) -> ServedDoc:
    """Serve a whitelisted figure asset from the doc's ``{stem}_figures/`` dir.

    Path-traversal-safe: the slug is validated, ``doc`` is whitelisted against the
    inventory stems (locating the doc's figures dir), ``path`` is pre-rejected (no
    ``..``/absolute/NUL/backslash) BEFORE any fs read, the extension must be in
    the shared ``_ASSET_CONTENT_TYPES`` whitelist (NEVER ``.html`` / active
    types), and the resolved file is commonpath-contained under the figures dir
    (catches symlink escapes).

    Raises ``SourceMaterialsError`` (403 disabled, 422 bad path / disallowed
    type, 404 unknown course / doc / figures-dir / member).
    """
    root = Path(libv2_root) if libv2_root is not None else _libv2_root()
    course_dir = _validate_slug(slug, root)
    if not is_enabled(course_dir):
        raise SourceMaterialsError(
            403, "source_materials_disabled", "source materials are disabled for this deployment"
        )
    path = _reject_asset_path(path)
    ext = Path(path).suffix.lower()
    ctype = _ASSET_CONTENT_TYPES.get(ext)
    if ctype is None:
        raise SourceMaterialsError(
            422, "asset_type_not_allowed", f"asset type not allowed: {ext or '(none)'}"
        )

    _html_path, root_dir = _resolve_doc_path(course_dir, doc)
    figures_dir = (root_dir / f"{doc}_figures").resolve()
    if not figures_dir.is_dir():
        raise SourceMaterialsError(404, "figures_missing", f"no figures dir for doc: {doc!r}")

    target = (figures_dir / path).resolve()
    # Post-resolution commonpath containment under the figures dir.
    try:
        common = os.path.commonpath([str(target), str(figures_dir)])
    except ValueError:
        raise SourceMaterialsError(
            422, "invalid_asset_path", f"asset escapes figures dir: {path!r}"
        ) from None
    if common != str(figures_dir):
        raise SourceMaterialsError(
            422, "invalid_asset_path", f"asset escapes figures dir: {path!r}"
        )
    if not target.is_file():
        raise SourceMaterialsError(404, "asset_missing", f"no figure asset: {path!r}")

    try:
        data = target.read_bytes()
    except OSError as exc:
        raise SourceMaterialsError(500, "asset_read_error", str(exc)) from exc

    return ServedDoc(body=data, media_type=ctype, headers=dict(_ASSET_HEADERS))


__all__ = [
    "ServedDoc",
    "SourceMaterialsError",
    "inventory",
    "is_enabled",
    "list_source_materials",
    "serve_source_doc",
    "serve_source_doc_asset",
]

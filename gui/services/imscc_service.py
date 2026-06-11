"""Studio IMSCC viewer service — archived-course Library + manifest + page/asset.

Backs the end-user "Studio" surface (``gui.routers.library`` → ``/api/library``,
``/api/courses/{id}/manifest|page|asset``). It is the read-only serving wrapper
over a course's packaged IMS Common Cartridge living at
``LibV2/courses/<slug>/source/imscc/*.imscc`` (the ``_IMSCC_ROOTS`` convention
of ``lib.retrieval.citation_anchor``).

What it factors / reuses (no reimplementation):

* **Org-tree walk** — the recursive ``<organizations>/<organization>/<item>``
  parse from ``Courseforge/scripts/imscc-extractor/imscc_extractor.py``
  (``_parse_organization`` / ``_parse_resources``), ported to read the manifest
  *in memory* from the cartridge zip (no extract-to-disk) so the viewer never
  unpacks untrusted archives. The depth→item_type heuristic matches the
  extractor (``module`` / ``unit`` / ``page``).
* **Sanitize + CSP serving** — the active-content scrub from
  ``gui.services.source_page.sanitize_soup`` (the shared, audited element/
  attribute stripper) plus the same restrictive ``Content-Security-Policy`` /
  ``X-Content-Type-Options: nosniff`` posture. Archived cartridge HTML is
  UNTRUSTED (it can carry ``<script>``, inline handlers, ``javascript:`` URLs).

Path-safety (closes the zip-member traversal hole): a learner-supplied
``item`` (resource identifier) is mapped to a member name *only* via the
manifest resource table — never used as a raw path. The asset path is
pre-rejected (no ``..`` segment, no absolute, no NUL/backslash) before any zip
read, mirroring ``source_page._reject_traversal``; zip member names are matched
against the cartridge's own ``namelist()`` so a fabricated member never reads
off-archive.

No network, no LLM, no decision capture — deterministic, read-only. Heavy
imports (bs4) are deferred to call time; ``lib.retrieval`` is NOT imported (the
org walk is self-contained here, single zip open per request).
"""

from __future__ import annotations

import os
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the audited source-viewer slug + sanitiser + error type.
from gui.services.source_page import SourcePageError, heading_slug, sanitize_soup

# --------------------------------------------------------------------------- #
# Constants (mirrors source_page + citation_anchor conventions)
# --------------------------------------------------------------------------- #

# Course-slug shape (identical to source_page._SLUG_RE).
_SLUG_RE = re.compile(r"^[a-z0-9][a-zA-Z0-9._-]*$")

# Where a course's packaged cartridge lives (citation_anchor._IMSCC_ROOTS).
_IMSCC_ROOTS = ("source/imscc",)

_DRIVE_RE = re.compile(r"^[A-Za-z]:")

# Restrictive CSP for the page HTML (no scripts; the viewer's own shim runs in
# the parent document, injected by the iframe-sandbox host — NOT served here).
# Identical posture to source_page._CSP.
_PAGE_CSP = "default-src 'none'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"

_PAGE_HEADERS: Dict[str, str] = {
    "Content-Security-Policy": _PAGE_CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

# Whitelisted asset extensions → content-type. Archived static assets only;
# never HTML (pages go through the sanitised page path) and never active types.
_ASSET_CONTENT_TYPES: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".css": "text/css; charset=utf-8",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".ico": "image/x-icon",
}

# Asset CSP: SVG can carry script, so serve assets with a no-active-content CSP
# + nosniff too (defence in depth even on whitelisted types).
_ASSET_HEADERS: Dict[str, str] = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


# --------------------------------------------------------------------------- #
# Public result / error types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ServedContent:
    """A served page or asset + the headers the route must echo back."""

    body: bytes
    media_type: str
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestItem:
    """One node in the course organization tree (mirrors the extractor shape)."""

    identifier: str
    title: str
    item: Optional[str] = None  # resource identifier (leaf pages only)
    href: Optional[str] = None  # member name inside the cartridge (leaf pages)
    item_type: str = "module"
    children: List["ManifestItem"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "title": self.title,
            "item": self.item,
            "href": self.href,
            "item_type": self.item_type,
            "children": [c.to_dict() for c in self.children],
        }


# IMSCCServiceError is just SourcePageError so the router maps both viewer paths
# through one (status, code, detail) contract.
IMSCCServiceError = SourcePageError


# --------------------------------------------------------------------------- #
# Course / cartridge resolution (path-safe)
# --------------------------------------------------------------------------- #


def _libv2_root() -> Path:
    """Resolve the LibV2 root, honoring ``ED4ALL_LIBV2_ROOT`` (call-time read)."""
    override = os.environ.get("ED4ALL_LIBV2_ROOT")
    if override:
        return Path(override)
    from lib.paths import LIBV2_PATH  # noqa: PLC0415

    return Path(LIBV2_PATH)


def _validate_slug(slug: str, libv2_root: Path) -> Path:
    """Validate the slug shape + course existence; return the resolved dir.

    Raises ``IMSCCServiceError`` (422 malformed, 404 unknown). Commonpath-checks
    the resolved course dir under ``courses/`` (defence in depth), mirroring
    ``source_page._validate_slug``.
    """
    if not slug or not _SLUG_RE.match(slug):
        raise IMSCCServiceError(422, "invalid_course_id", f"invalid course slug: {slug!r}")
    courses_dir = (Path(libv2_root) / "courses").resolve()
    course_dir = (courses_dir / slug).resolve()
    try:
        common = os.path.commonpath([str(course_dir), str(courses_dir)])
    except ValueError:
        raise IMSCCServiceError(
            422, "invalid_course_id", f"slug escapes courses root: {slug!r}"
        ) from None
    if common != str(courses_dir):
        raise IMSCCServiceError(
            422, "invalid_course_id", f"slug escapes courses root: {slug!r}"
        )
    if not course_dir.is_dir():
        raise IMSCCServiceError(404, "course_not_found", f"course not found: {slug}")
    return course_dir


def _find_cartridge(course_dir: Path) -> Optional[Path]:
    """Return the first ``*.imscc`` cartridge under the IMSCC roots, or None."""
    for root in _IMSCC_ROOTS:
        rdir = (course_dir / root)
        if not rdir.is_dir():
            continue
        cartridges = sorted(rdir.glob("*.imscc"))
        if cartridges:
            return cartridges[0]
    return None


def _open_cartridge(slug: str, libv2_root: Optional[Path] = None) -> Tuple[Path, Path]:
    """Resolve ``(course_dir, cartridge_path)`` for ``slug``; raise on miss.

    404 if the course dir or cartridge is absent (the Library only lists courses
    that DO carry a cartridge, but a direct API hit on a chunks-only course must
    404 honestly rather than 500).
    """
    root = Path(libv2_root) if libv2_root is not None else _libv2_root()
    course_dir = _validate_slug(slug, root)
    cartridge = _find_cartridge(course_dir)
    if cartridge is None:
        raise IMSCCServiceError(
            404, "cartridge_not_found", f"no IMSCC cartridge archived for course: {slug}"
        )
    return course_dir, cartridge


# --------------------------------------------------------------------------- #
# Manifest parse (org-tree walk, factored from imscc_extractor, in-memory)
# --------------------------------------------------------------------------- #


def _read_manifest(zf: zipfile.ZipFile) -> Tuple[object, str]:
    """Return ``(manifest_root, manifest_prefix)`` parsed from the cartridge zip.

    ``manifest_prefix`` is the directory the manifest lives in (``""`` at root,
    or ``"inner/"`` for nested cartridges) so resource hrefs resolve to the
    correct zip member. Raises ``IMSCCServiceError(500, ...)`` on a missing /
    unparseable manifest.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    names = zf.namelist()
    manifest_member = None
    for n in names:
        if n == "imsmanifest.xml" or n.endswith("/imsmanifest.xml"):
            # Prefer the shallowest manifest (root over a nested copy).
            if manifest_member is None or n.count("/") < manifest_member.count("/"):
                manifest_member = n
    if manifest_member is None:
        raise IMSCCServiceError(
            500, "manifest_missing", "imsmanifest.xml not found in cartridge"
        )
    prefix = manifest_member[: manifest_member.rfind("/") + 1] if "/" in manifest_member else ""
    try:
        root = ET.fromstring(zf.read(manifest_member))
    except ET.ParseError as exc:
        raise IMSCCServiceError(500, "manifest_parse_error", f"bad manifest XML: {exc}") from exc
    return root, prefix


def _parse_resources(root) -> Dict[str, str]:  # noqa: ANN001 — ET.Element
    """Map ``resource identifier -> href`` (factored from extractor._parse_resources)."""
    out: Dict[str, str] = {}
    resources_elem = root.find(".//{*}resources")
    if resources_elem is None:
        return out
    for res in resources_elem.findall(".//{*}resource"):
        identifier = res.get("identifier", "")
        href = res.get("href", "")
        if not href:
            file_el = res.find("./{*}file")
            if file_el is not None:
                href = file_el.get("href", "")
        if identifier:
            out[identifier] = href
    return out


def _parse_organization(root, resources: Dict[str, str]) -> List[ManifestItem]:  # noqa: ANN001
    """Recursive org-tree walk (factored from extractor._parse_organization).

    Mirrors the extractor's default-org selection + depth→item_type heuristic
    (``module`` at depth 0, ``unit`` at depth 1, ``page`` for any item that
    points at a resource). Leaf pages carry the resolved cartridge member as
    ``href`` so the page endpoint can read it directly.
    """
    orgs = root.find(".//{*}organizations")
    if orgs is None:
        return []
    default_org_id = orgs.get("default", "")
    org_elem = None
    for org in orgs.findall(".//{*}organization"):
        if org.get("identifier") == default_org_id:
            org_elem = org
            break
    if org_elem is None:
        org_elem = orgs.find(".//{*}organization")
    if org_elem is None:
        return []

    def parse_item(elem, depth: int = 0) -> ManifestItem:  # noqa: ANN001
        identifier = elem.get("identifier", "")
        identifierref = elem.get("identifierref", "")
        title_elem = elem.find("./{*}title")
        title = (
            title_elem.text.strip()
            if title_elem is not None and title_elem.text
            else (identifier or "Untitled")
        )
        if depth == 0:
            item_type = "module"
        elif depth == 1:
            item_type = "unit"
        elif identifierref:
            item_type = "page"
        else:
            item_type = "module"
        href = resources.get(identifierref) if identifierref else None
        node = ManifestItem(
            identifier=identifier,
            title=title,
            item=identifierref or None,
            href=href or None,
            item_type=item_type if not identifierref else "page",
            children=[],
        )
        children = [parse_item(c, depth + 1) for c in elem.findall("./{*}item")]
        # ManifestItem is frozen; rebuild with children.
        return ManifestItem(
            identifier=node.identifier,
            title=node.title,
            item=node.item,
            href=node.href,
            item_type=node.item_type,
            children=children,
        )

    # Some cartridges wrap the whole course under a single ROOT item; flatten
    # one level so the tree's top entries are the real modules (matches the
    # extractor's top-level item iteration).
    top_items = org_elem.findall("./{*}item")
    parsed = [parse_item(it, 0) for it in top_items]
    if len(parsed) == 1 and parsed[0].children and parsed[0].href is None:
        # Re-parse the single ROOT's children as the top level (depth resets).
        only_root = top_items[0]
        parsed = [parse_item(c, 0) for c in only_root.findall("./{*}item")]
    return parsed


def _flatten_pages(items: List[ManifestItem]) -> List[ManifestItem]:
    """Depth-first list of leaf page items (those carrying a resource ``item``)."""
    out: List[ManifestItem] = []
    for it in items:
        if it.item and it.href:
            out.append(it)
        out.extend(_flatten_pages(it.children))
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Disk-usage cache (D5)
# --------------------------------------------------------------------------- #
#
# Courses are large (corpus + chunks + vector index + source PDFs), so an
# os.scandir walk per card per Library load is wasteful. Cache the per-course
# byte total with a short TTL keyed by the resolved course path. Simple and
# self-healing: a stale entry only over-reports for at most _DISK_CACHE_TTL_S
# seconds, and a removed course drops out of the listing entirely.
_DISK_CACHE_TTL_S = 60.0
_disk_cache: Dict[str, Tuple[float, int]] = {}


def _cached_disk_bytes(course_dir: Path) -> int:
    """Return the course dir's recursive byte size, TTL-cached per path."""
    from LibV2.tools.libv2.remove import dir_size_bytes  # noqa: PLC0415

    key = str(course_dir)
    now = time.monotonic()
    hit = _disk_cache.get(key)
    if hit is not None and (now - hit[0]) < _DISK_CACHE_TTL_S:
        return hit[1]
    size = dir_size_bytes(course_dir)
    _disk_cache[key] = (now, size)
    return size


def _invalidate_disk_cache(course_dir: Optional[Path] = None) -> None:
    """Drop a course's cached size (or the whole cache when ``None``)."""
    if course_dir is None:
        _disk_cache.clear()
    else:
        _disk_cache.pop(str(course_dir), None)


def list_library(libv2_root: Optional[Path] = None) -> List[dict]:
    """Course cards for every archived course that carries an IMSCC cartridge.

    Card shape: ``{slug, title, page_count, has_vector_index, disk_bytes}``.
    Courses are discovered dynamically by scanning ``LibV2/courses/*`` — never
    hardcoded. A course with no cartridge is omitted (the Studio Library is the
    *viewable* set). Best-effort per course: a malformed cartridge degrades to a
    card with ``page_count=0`` rather than dropping the course or raising.
    ``disk_bytes`` is the recursive on-disk size (TTL-cached per course path).
    """
    root = Path(libv2_root) if libv2_root is not None else _libv2_root()
    courses_dir = root / "courses"
    if not courses_dir.is_dir():
        return []
    cards: List[dict] = []
    for entry in sorted(courses_dir.iterdir()):
        if not entry.is_dir() or not _SLUG_RE.match(entry.name):
            continue
        cartridge = _find_cartridge(entry)
        if cartridge is None:
            continue
        title = entry.name
        page_count = 0
        try:
            with zipfile.ZipFile(cartridge) as zf:
                mroot, _ = _read_manifest(zf)
                title = _manifest_title(mroot) or entry.name
                resources = _parse_resources(mroot)
                org = _parse_organization(mroot, resources)
                page_count = len(_flatten_pages(org))
        except (IMSCCServiceError, zipfile.BadZipFile, OSError):
            pass
        cards.append(
            {
                "slug": entry.name,
                "title": title,
                "page_count": page_count,
                "has_vector_index": (entry / "vector_index").is_dir(),
                "disk_bytes": _cached_disk_bytes(entry),
            }
        )
    return cards


def _manifest_title(root) -> str:  # noqa: ANN001 — ET.Element
    """Best-effort course title from manifest metadata (lom/general/title)."""
    for path in (
        ".//{*}lom//{*}general//{*}title//{*}string",
        ".//{*}general//{*}title//{*}string",
        ".//{*}title//{*}string",
        ".//{*}title",
    ):
        el = root.find(path)
        if el is not None and el.text and el.text.strip():
            return el.text.strip()
    return ""


def get_manifest(slug: str, libv2_root: Optional[Path] = None) -> dict:
    """Return the course org tree as JSON (``{slug, title, items: [...]}``).

    Raises ``IMSCCServiceError`` (404 unknown course / no cartridge, 500 bad
    manifest).
    """
    _course_dir, cartridge = _open_cartridge(slug, libv2_root)
    with zipfile.ZipFile(cartridge) as zf:
        mroot, _prefix = _read_manifest(zf)
        title = _manifest_title(mroot) or slug
        resources = _parse_resources(mroot)
        items = _parse_organization(mroot, resources)
    return {
        "slug": slug,
        "title": title,
        "items": [it.to_dict() for it in items],
    }


def _resolve_member(zf: zipfile.ZipFile, href: str, prefix: str) -> Optional[str]:
    """Map a manifest href to an actual zip member (exact, prefixed, basename)."""
    names = set(zf.namelist())
    for cand in (href, prefix + href):
        if cand in names:
            return cand
    base = Path(href).name
    for n in zf.namelist():
        if Path(n).name == base:
            return n
    return None


def get_page(slug: str, item: str, libv2_root: Optional[Path] = None) -> ServedContent:
    """Serve one cartridge page (resolved via the manifest resource id), sanitised.

    ``item`` is a RESOURCE IDENTIFIER from the manifest tree — never a raw path.
    It is looked up in the resource table; the resolved href is matched against
    the cartridge's own member list. The page HTML is run through the shared
    ``sanitize_soup`` scrub (drops ``<script>`` / inline handlers / active URLs),
    heading ids are injected for in-page anchors, and ``<html lang>`` is ensured.
    Returns the sanitised HTML + restrictive CSP / nosniff headers.

    Raises ``IMSCCServiceError`` (422 empty item, 404 unknown course / cartridge
    / item, 500 manifest/sanitise failure).
    """
    if not item or not item.strip():
        raise IMSCCServiceError(422, "invalid_item", "item (resource id) is required")
    _course_dir, cartridge = _open_cartridge(slug, libv2_root)
    with zipfile.ZipFile(cartridge) as zf:
        mroot, prefix = _read_manifest(zf)
        resources = _parse_resources(mroot)
        href = resources.get(item)
        if not href:
            raise IMSCCServiceError(404, "item_not_found", f"unknown manifest item: {item!r}")
        member = _resolve_member(zf, href, prefix)
        if member is None:
            raise IMSCCServiceError(
                404, "page_missing", f"no cartridge member for item: {item!r}"
            )
        try:
            raw = zf.read(member).decode("utf-8", errors="replace")
        except (KeyError, OSError) as exc:
            raise IMSCCServiceError(500, "page_read_error", str(exc)) from exc

    html = _sanitize_page(raw)
    return ServedContent(
        body=html.encode("utf-8"),
        media_type="text/html; charset=utf-8",
        headers=dict(_PAGE_HEADERS),
    )


def _sanitize_page(raw_html: str) -> str:
    """Scrub active content + inject heading ids + ensure lang (page serving)."""
    from bs4 import BeautifulSoup  # noqa: PLC0415

    soup = BeautifulSoup(raw_html, "html.parser")
    sanitize_soup(soup)  # shared audited element/attribute scrub
    # Inject heading ids (only where missing) for in-page anchor navigation.
    for level in ("h1", "h2", "h3", "h4", "h5", "h6"):
        for heading in soup.find_all(level):
            if heading.get("id"):
                continue
            text = heading.get_text(" ", strip=True)
            slug_id = heading_slug(text)
            if slug_id:
                heading["id"] = slug_id
    html_el = soup.find("html")
    if html_el is not None and not html_el.get("lang"):
        html_el["lang"] = "en"
        html_el["data-lang-injected"] = "true"
    return str(soup)


def _reject_asset_path(path: str) -> str:
    """Pre-reject a hostile asset path BEFORE any zip read (mirrors source_page)."""
    if not path or not path.strip():
        raise IMSCCServiceError(422, "invalid_asset_path", "path is required")
    if "\x00" in path:
        raise IMSCCServiceError(422, "invalid_asset_path", "path contains a NUL byte")
    if "\\" in path:
        raise IMSCCServiceError(422, "invalid_asset_path", "path contains a backslash")
    if path.startswith("/") or _DRIVE_RE.match(path):
        raise IMSCCServiceError(422, "invalid_asset_path", f"absolute path rejected: {path!r}")
    if Path(path).is_absolute():
        raise IMSCCServiceError(422, "invalid_asset_path", f"absolute path rejected: {path!r}")
    if any(seg == ".." for seg in path.split("/")):
        raise IMSCCServiceError(422, "invalid_asset_path", f"path escapes cartridge: {path!r}")
    return path


def get_asset(slug: str, path: str, libv2_root: Optional[Path] = None) -> ServedContent:
    """Serve a whitelisted static asset (image/css/font) from the cartridge.

    Path-traversal-safe: the path is pre-rejected (no ``..``/absolute/NUL/
    backslash) BEFORE any zip read, the extension must be in the content-type
    whitelist (NEVER ``.html`` — pages route through ``get_page``; never an
    active type), and the requested member must literally exist in the
    cartridge's ``namelist()`` (a fabricated member never reads off-archive).

    Raises ``IMSCCServiceError`` (422 bad path / disallowed type, 404 unknown
    course / cartridge / member).
    """
    path = _reject_asset_path(path)
    ext = Path(path).suffix.lower()
    ctype = _ASSET_CONTENT_TYPES.get(ext)
    if ctype is None:
        raise IMSCCServiceError(
            422, "asset_type_not_allowed", f"asset type not allowed: {ext or '(none)'}"
        )
    _course_dir, cartridge = _open_cartridge(slug, libv2_root)
    with zipfile.ZipFile(cartridge) as zf:
        names = set(zf.namelist())
        member = None
        if path in names:
            member = path
        else:
            # Allow a manifest-prefixed lookup (nested cartridge), basename last.
            base = Path(path).name
            for n in zf.namelist():
                if n == path or n.endswith("/" + path) or Path(n).name == base:
                    if Path(n).suffix.lower() == ext:
                        member = n
                        break
        if member is None:
            raise IMSCCServiceError(404, "asset_missing", f"no cartridge asset: {path!r}")
        try:
            data = zf.read(member)
        except (KeyError, OSError) as exc:
            raise IMSCCServiceError(500, "asset_read_error", str(exc)) from exc
    return ServedContent(
        body=data,
        media_type=ctype,
        headers=dict(_ASSET_HEADERS),
    )


__all__ = [
    "ServedContent",
    "ManifestItem",
    "IMSCCServiceError",
    "list_library",
    "get_manifest",
    "get_page",
    "get_asset",
]

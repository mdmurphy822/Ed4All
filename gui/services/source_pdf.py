"""Source-PDF serving — the final hop of the B4 provenance chain.

A cited answer traces chunk → course page → DART source block → original PDF
page. The first three hops already resolve (citation anchor + ``source_page`` /
``imscc_service``); this module serves the LAST hop: the archived raw PDF that a
``dart:{slug}#{block_id}`` source block was synthesized from.

Layout (set by ``MCP/tools/pipeline_tools.py::archive_to_libv2``): raw PDFs land
under ``LibV2/courses/<slug>/source/pdf/<name>.pdf``. The DART converter writes a
``{stem}_synthesized.json`` sidecar carrying a ``source_pdf`` field that records
which PDF a staged HTML document (and therefore its ``dart:{slug}`` sourceId
half) came from; those sidecars are archived alongside the course, so this module
resolves a sourceId-derived ``file`` hint to the real archived PDF basename via
them when present.

Security posture (mirrors ``source_page`` / ``imscc_service``):

* the course slug is regex-validated + commonpath-contained under ``courses/``;
* the requested ``file`` is matched ONLY against the basenames actually present
  in the course's source-PDF directory (a whitelist from the dir listing) — it
  is NEVER used as a path; traversal, absolute paths, and unknown names are
  rejected before any fs read;
* a course with no archived PDFs 404s with an explanation rather than guessing.

Single-page extraction: when ``pypdf`` or ``PyMuPDF`` (``fitz``) is importable we
extract just the requested page and serve it as a one-page ``application/pdf``.
When neither is available (the ``[dart]`` PDF extras aren't installed) we fall
back to serving the WHOLE PDF and let the browser deep-link to ``#page=N`` via
the response — the acceptable degradation documented in the B4 brief. Either way
the response is ``application/pdf``; the caller opens it in a new tab.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Course-slug shape (identical to source_page / imscc_service).
_SLUG_RE = re.compile(r"^[a-z0-9][a-zA-Z0-9._-]*$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")

# Where archive_to_libv2 lands raw PDFs (and a couple of tolerant alternates a
# hand-archived corpus might use).
_PDF_ROOTS = ("source/pdf", "source/pdfs", "raw")


class SourcePdfError(Exception):
    """Typed error → the router maps ``(status, code, detail)`` to an HTTP body."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class SourcePdfResult:
    """A served source-PDF (single page or whole-file fallback)."""

    body: bytes
    media_type: str
    headers: Dict[str, str]


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def _libv2_root() -> Path:
    """Resolve the LibV2 root, honoring ``ED4ALL_LIBV2_ROOT`` (call-time read)."""
    override = os.environ.get("ED4ALL_LIBV2_ROOT")
    if override:
        return Path(override)
    from lib.paths import LIBV2_PATH  # noqa: PLC0415

    return Path(LIBV2_PATH)


def _validate_slug(slug: str, libv2_root: Path) -> Path:
    """Validate slug shape + course existence; return the resolved course dir."""
    if not slug or not _SLUG_RE.match(slug):
        raise SourcePdfError(422, "invalid_course_id", f"invalid course slug: {slug!r}")
    courses_dir = (Path(libv2_root) / "courses").resolve()
    course_dir = (courses_dir / slug).resolve()
    try:
        common = os.path.commonpath([str(course_dir), str(courses_dir)])
    except ValueError:
        raise SourcePdfError(
            422, "invalid_course_id", f"slug escapes courses root: {slug!r}"
        ) from None
    if common != str(courses_dir):
        raise SourcePdfError(422, "invalid_course_id", f"slug escapes courses root: {slug!r}")
    if not course_dir.is_dir():
        raise SourcePdfError(404, "course_not_found", f"course not found: {slug}")
    return course_dir


def _pdf_dir(course_dir: Path) -> Optional[Path]:
    """Return the first source-PDF root that exists for the course, or ``None``."""
    for root in _PDF_ROOTS:
        cand = course_dir / root
        if cand.is_dir():
            return cand
    return None


def _list_pdfs(pdf_dir: Path) -> List[Path]:
    """Whitelist: every ``*.pdf`` directly under the source-PDF dir."""
    return sorted(p for p in pdf_dir.glob("*.pdf") if p.is_file())


def _slugify_pdf_name(name: str) -> str:
    """Gentle slug of a PDF basename, mirroring DART's ``dart:{slug}`` rule.

    ``path.stem.lower().replace(" ", "-")`` — underscores/hyphens/digits pass
    through unchanged so a sourceId slug can be matched against a PDF filename.
    """
    return Path(name).stem.lower().replace(" ", "-")


def _sidecar_source_pdf_map(course_dir: Path) -> Dict[str, str]:
    """Map a DART document slug → archived PDF basename via archived sidecars.

    Walks ``*_synthesized.json`` sidecars (archived under the course's source
    dirs) and keys their ``source_pdf`` basename by the sidecar's own ``slug``
    (and, defensively, the sidecar filename stem with ``_synthesized`` stripped).
    Returns ``{}`` when no sidecars are present — the caller then falls back to
    matching the sourceId slug directly against PDF filenames.
    """
    out: Dict[str, str] = {}
    for sidecar in course_dir.rglob("*_synthesized.json"):
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        src_pdf = data.get("source_pdf")
        if not isinstance(src_pdf, str) or not src_pdf.strip():
            continue
        pdf_name = Path(src_pdf).name
        keys = set()
        slug = data.get("slug")
        if isinstance(slug, str) and slug:
            keys.add(slug.lower())
        stem = sidecar.stem
        if stem.endswith("_synthesized"):
            stem = stem[: -len("_synthesized")]
        keys.add(_slugify_pdf_name(stem + ".x"))
        for k in keys:
            out.setdefault(k, pdf_name)
    return out


def _resolve_pdf(
    file_hint: str, pdf_dir: Path, course_dir: Path
) -> Tuple[Path, Optional[str]]:
    """Resolve a ``file`` hint to an archived PDF path (whitelisted).

    ``file_hint`` is the basename or sourceId-slug carried by the citation's
    ``source_block`` (``dart:{slug}#...`` → ``{slug}``). Resolution order:

    1. exact basename match against the dir listing (``foo.pdf`` or ``foo``);
    2. the sidecar ``source_pdf`` map (sourceId slug → archived PDF basename);
    3. sourceId-slug match against each PDF's slugified stem;
    4. single-PDF fallback: when the course archived exactly one PDF, serve it
       (the common single-textbook corpus).

    Returns ``(pdf_path, warning_or_None)``. Raises ``SourcePdfError`` only when
    no PDF can be resolved AND the course has more than one (ambiguous → omit,
    never guess wrong). The warning is surfaced as a response header for
    debuggability when a fallback was taken.
    """
    pdfs = _list_pdfs(pdf_dir)
    if not pdfs:
        raise SourcePdfError(
            404, "no_source_pdf", "no source PDFs archived for this course"
        )
    by_name = {p.name.lower(): p for p in pdfs}
    by_stem = {_slugify_pdf_name(p.name): p for p in pdfs}

    hint = (file_hint or "").strip()
    if hint:
        # Reject a hostile hint before any matching (defence in depth — we only
        # ever match basenames, but a slash/dotdot hint should never resolve).
        if "/" in hint or "\\" in hint or ".." in hint or "\x00" in hint:
            raise SourcePdfError(
                422, "invalid_file", f"invalid source-pdf file: {file_hint!r}"
            )
        low = hint.lower()
        # 1. exact basename (with or without .pdf).
        if low in by_name:
            return by_name[low], None
        if (low + ".pdf") in by_name:
            return by_name[low + ".pdf"], None
        # 2. sidecar source_pdf map.
        sidecar_map = _sidecar_source_pdf_map(course_dir)
        mapped = sidecar_map.get(low)
        if mapped and mapped.lower() in by_name:
            return by_name[mapped.lower()], None
        # 3. slugified-stem match.
        slug_hint = _slugify_pdf_name(hint + ".x")
        if slug_hint in by_stem:
            return by_stem[slug_hint], None
        if low in by_stem:
            return by_stem[low], None

    # 4. single-PDF fallback.
    if len(pdfs) == 1:
        return pdfs[0], "single_pdf_fallback"

    # Ambiguous multi-PDF corpus with no resolvable hint — never guess.
    raise SourcePdfError(
        404,
        "source_pdf_unresolved",
        "source PDF mapping unavailable for this citation (multi-PDF course; "
        f"hint={file_hint!r} matched none of {sorted(by_name)})",
    )


# --------------------------------------------------------------------------- #
# Single-page extraction (graceful: whole-file fallback when no PDF lib)
# --------------------------------------------------------------------------- #


def _extract_single_page(pdf_bytes: bytes, page: int) -> Optional[bytes]:
    """Return a one-page PDF for 1-indexed ``page``; ``None`` if no PDF lib.

    Tries ``pypdf`` first (the lightest), then ``PyMuPDF`` (``fitz``, the
    declared ``[dart]`` extra). A page index out of range raises
    ``SourcePdfError`` (the citation pointed past the document). ``None`` signals
    the caller to fall back to serving the whole PDF.
    """
    # pypdf path.
    try:
        from pypdf import PdfReader, PdfWriter  # noqa: PLC0415
        import io  # noqa: PLC0415

        reader = PdfReader(io.BytesIO(pdf_bytes))
        n = len(reader.pages)
        if page < 1 or page > n:
            raise SourcePdfError(
                404, "page_out_of_range", f"page {page} not in document (1..{n})"
            )
        writer = PdfWriter()
        writer.add_page(reader.pages[page - 1])
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()
    except SourcePdfError:
        raise
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 — a corrupt PDF / odd pypdf state → fall back
        return None

    # PyMuPDF (fitz) path.
    try:
        import fitz  # noqa: PLC0415
    except ImportError:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:  # noqa: BLE001 — won't open → fall back to whole-file
        return None
    try:
        n = doc.page_count
        if page < 1 or page > n:
            raise SourcePdfError(
                404, "page_out_of_range", f"page {page} not in document (1..{n})"
            )
        out = fitz.open()
        out.insert_pdf(doc, from_page=page - 1, to_page=page - 1)
        data = out.tobytes()
        out.close()
        return data
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def serve_source_pdf_page(
    course_id: str,
    file: str,
    page: int,
    *,
    libv2_root: Optional[Path] = None,
) -> SourcePdfResult:
    """Serve a single archived-PDF page (or the whole PDF as a fallback).

    Raises ``SourcePdfError`` for an invalid slug/file, a course with no archived
    PDFs, an unresolvable multi-PDF mapping, or an out-of-range page. The body is
    always ``application/pdf``; ``X-Source-Pdf-Mode`` distinguishes
    ``single-page`` from ``whole-file`` so the client (and tests) can tell which
    path was taken.

    ``page=0`` (or an absent/negative page) means *whole-file* mode: the entire
    archived PDF is served inline (the browse-surface "open the original PDF"
    affordance), distinct from the per-page citation hop. Source-PDF serving is
    gated under the same ``ED4ALL_SOURCE_MATERIALS`` operator toggle as the rest
    of the source-materials surface — a deliberate behavior change (default-on,
    so no change for default deployments).
    """
    root = Path(libv2_root) if libv2_root is not None else _libv2_root()
    course_dir = _validate_slug(course_id, root)

    # Operator toggle: fold source-PDF serving under the shared source-materials
    # gate (§2.5). 403 when disabled rather than serving the verbatim textbook.
    try:
        from gui.services.source_materials import is_enabled as _sm_enabled  # noqa: PLC0415

        if not _sm_enabled(course_dir):
            raise SourcePdfError(
                403, "source_materials_disabled",
                "source materials are disabled for this deployment",
            )
    except ImportError:
        # source_materials always ships with source_pdf in the gui extra; if it
        # somehow can't import, fail open to the prior (ungated) behavior.
        pass

    try:
        page_num = int(page)
    except (TypeError, ValueError):
        raise SourcePdfError(422, "invalid_page", f"invalid page: {page!r}") from None
    if page_num < 0:
        raise SourcePdfError(422, "invalid_page", f"page must be >= 0, got {page_num}")
    # page == 0 → whole-file mode (additive; the per-page citation hop passes >=1).
    whole_file = page_num == 0

    pdf_dir = _pdf_dir(course_dir)
    if pdf_dir is None:
        raise SourcePdfError(
            404, "no_source_pdf", "no source PDFs archived for this course"
        )

    pdf_path, warning = _resolve_pdf(file, pdf_dir, course_dir)
    try:
        pdf_bytes = pdf_path.read_bytes()
    except OSError as exc:
        raise SourcePdfError(500, "pdf_read_failed", str(exc)) from exc

    headers: Dict[str, str] = {
        "Content-Disposition": f'inline; filename="{pdf_path.name}"',
        "X-Content-Type-Options": "nosniff",
    }
    if warning:
        headers["X-Source-Pdf-Warning"] = warning

    # Whole-file mode (page<=0): serve the entire archived PDF inline (the browse
    # affordance), skipping single-page extraction entirely.
    if whole_file:
        headers["X-Source-Pdf-Mode"] = "whole-file"
        headers["X-Source-Pdf-Page"] = "0"
        return SourcePdfResult(body=pdf_bytes, media_type="application/pdf", headers=headers)

    single = _extract_single_page(pdf_bytes, page_num)
    if single is not None:
        headers["X-Source-Pdf-Mode"] = "single-page"
        headers["X-Source-Pdf-Page"] = str(page_num)
        return SourcePdfResult(body=single, media_type="application/pdf", headers=headers)

    # Fallback: serve the whole PDF; the browser deep-links to #page=N.
    headers["X-Source-Pdf-Mode"] = "whole-file"
    headers["X-Source-Pdf-Page"] = str(page_num)
    return SourcePdfResult(body=pdf_bytes, media_type="application/pdf", headers=headers)


__all__ = [
    "SourcePdfError",
    "SourcePdfResult",
    "serve_source_pdf_page",
]

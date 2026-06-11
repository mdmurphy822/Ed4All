"""FastAPI ``TestClient`` tests for the Studio Library + IMSCC viewer router.

Covers all four endpoints (``/api/library``, ``/api/courses/{id}/manifest|page|
asset``) over SYNTHETIC tmp-path course archives — a builder writes a tiny
IMS Common Cartridge into ``<libv2_root>/courses/<slug>/source/imscc/*.imscc``
(no real LibV2 course is touched; state isolates via the ``libv2_root`` fixture).

Asserts the happy path AND the security contract: path-traversal rejections on
the asset path, the content-type whitelist (no ``.html`` / active types as
assets), unknown-item / unknown-course 404s, and the sanitiser stripping
``<script>`` + inline handlers from served pages.

Skipped on a default install with no ``fastapi`` (the ``gui`` extra is opt-in).
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bs4")

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402

# Reuse the committed pure-Python PDF builder (no fitz/pypdf/reportlab needed).
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "pipeline"
if str(_FIXTURE_DIR) not in sys.path:
    sys.path.insert(0, str(_FIXTURE_DIR))
from build_fixture_pdf import build_pdf  # noqa: E402


def _tiny_pdf(n_pages: int = 3) -> bytes:
    """A valid multi-page PDF (one line of text per page)."""
    return build_pdf([[f"Page {i + 1} content."] for i in range(n_pages)])


def _has_pdf_lib() -> bool:
    """True when single-page extraction is available (pypdf or PyMuPDF)."""
    try:
        import pypdf  # noqa: F401,PLC0415

        return True
    except ImportError:
        pass
    try:
        import fitz  # noqa: F401,PLC0415

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Synthetic-cartridge builder fixture
# --------------------------------------------------------------------------- #

_MANIFEST = """<?xml version='1.0' encoding='utf-8'?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
          xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest"
          identifier="DEMO_manifest">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.3.0</schemaversion>
    <lomimscc:lom><lomimscc:general><lomimscc:title>
      <lomimscc:string language="en">Demo Course Title</lomimscc:string>
    </lomimscc:title></lomimscc:general></lomimscc:lom>
  </metadata>
  <organizations>
    <organization identifier="ORG_1" structure="rooted-hierarchy">
      <item identifier="ROOT">
        <title>Demo Course Title</title>
        <item identifier="WEEK_1">
          <title>Week 1</title>
          <item identifier="ITEM_OV" identifierref="RES_overview">
            <title>Overview</title>
          </item>
          <item identifier="ITEM_SC" identifierref="RES_self_check">
            <title>Self Check</title>
          </item>
        </item>
      </item>
    </organization>
  </organizations>
  <resources>
    <resource identifier="RES_overview" type="webcontent" href="week_01/overview.html">
      <file href="week_01/overview.html" />
    </resource>
    <resource identifier="RES_self_check" type="webcontent" href="week_01/self_check.html">
      <file href="week_01/self_check.html" />
    </resource>
  </resources>
</manifest>
"""

_OVERVIEW_HTML = """<!DOCTYPE html>
<html><head><title>Overview</title>
<link rel="stylesheet" href="assets/style.css"></head>
<body>
  <h1>Week 1 Overview</h1>
  <p>Intro to the topic.</p>
  <img src="assets/diagram.png" alt="A diagram">
</body></html>
"""

# A self-check page carrying the inert-component shapes the shim reanimates +
# the active content the sanitiser must strip.
_SELF_CHECK_HTML = """<!DOCTYPE html>
<html><head><title>Self Check</title>
<script>function toggleAnswer(id){document.getElementById(id).style.display='block';}</script>
</head>
<body onload="track()">
  <h1>Self Check</h1>
  <div class="self-check" data-cf-component="self-check">
    <button class="reveal-btn" aria-controls="ans1" aria-expanded="false"
            onclick="toggleAnswer('ans1')">Show Answer</button>
    <div id="ans1" class="answer-reveal" style="display:none"><p>The answer.</p></div>
  </div>
  <a href="javascript:alert(1)">bad link</a>
</body></html>
"""


def _build_cartridge(libv2_root: Path, slug: str, *, with_vector_index: bool = False) -> Path:
    """Write a tiny valid ``.imscc`` under ``courses/<slug>/source/imscc/``."""
    imscc_dir = libv2_root / "courses" / slug / "source" / "imscc"
    imscc_dir.mkdir(parents=True, exist_ok=True)
    cartridge = imscc_dir / f"{slug.upper()}.imscc"
    with zipfile.ZipFile(cartridge, "w") as zf:
        zf.writestr("imsmanifest.xml", _MANIFEST)
        zf.writestr("week_01/overview.html", _OVERVIEW_HTML)
        zf.writestr("week_01/self_check.html", _SELF_CHECK_HTML)
        zf.writestr("assets/style.css", "body{color:#111}")
        zf.writestr("assets/diagram.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        zf.writestr("secret.txt", "should never be served")
    if with_vector_index:
        (libv2_root / "courses" / slug / "vector_index").mkdir(parents=True, exist_ok=True)
    return cartridge


@pytest.fixture
def demo_course(libv2_root: Path):
    """One synthetic course ('demo-101') with a cartridge + a vector index."""
    _build_cartridge(libv2_root, "demo-101", with_vector_index=True)
    return "demo-101"


@pytest.fixture
def client(state_dir, libv2_root):
    """A TestClient over a fresh FULL-mode app with isolated state + LibV2."""
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# /api/library
# --------------------------------------------------------------------------- #


def test_library_lists_courses_with_cartridges(client, demo_course, libv2_root):
    # A second course WITHOUT a cartridge must NOT appear in the library.
    (libv2_root / "courses" / "no-cart-202" / "imscc_chunks").mkdir(parents=True)
    resp = client.get("/api/library")
    assert resp.status_code == 200
    cards = resp.json()
    slugs = {c["slug"] for c in cards}
    assert "demo-101" in slugs
    assert "no-cart-202" not in slugs
    card = next(c for c in cards if c["slug"] == "demo-101")
    assert card["title"] == "Demo Course Title"
    assert card["page_count"] == 2
    assert card["has_vector_index"] is True


def test_library_empty_when_no_courses(client):
    resp = client.get("/api/library")
    assert resp.status_code == 200
    assert resp.json() == []


def test_library_card_carries_disk_bytes(client, demo_course):
    resp = client.get("/api/library")
    assert resp.status_code == 200
    card = next(c for c in resp.json() if c["slug"] == "demo-101")
    assert "disk_bytes" in card
    assert isinstance(card["disk_bytes"], int)
    assert card["disk_bytes"] > 0, "a real cartridge course has a non-zero on-disk size"


# --------------------------------------------------------------------------- #
# DELETE /api/library/{course_id}  (D5 destructive removal)
# --------------------------------------------------------------------------- #


def test_delete_requires_confirm_echo(client, demo_course, libv2_root):
    # No confirm param → 400, course untouched.
    resp = client.request("DELETE", f"/api/library/{demo_course}")
    assert resp.status_code == 400
    assert resp.json()["error"] == "confirm_required"
    assert (libv2_root / "courses" / demo_course).exists()


def test_delete_wrong_confirm_echo_400(client, demo_course, libv2_root):
    resp = client.request("DELETE", f"/api/library/{demo_course}?confirm=wrong-slug")
    assert resp.status_code == 400
    assert resp.json()["error"] == "confirm_required"
    assert (libv2_root / "courses" / demo_course).exists()


def test_delete_success_removes_course(client, demo_course, libv2_root):
    resp = client.request(
        "DELETE", f"/api/library/{demo_course}?confirm={demo_course}"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["removed"] is True
    assert body["slug"] == demo_course
    assert not (libv2_root / "courses" / demo_course).exists()
    # The course drops out of the library listing.
    listing = client.get("/api/library").json()
    assert demo_course not in {c["slug"] for c in listing}


def test_delete_missing_course_404(client, libv2_root):
    (libv2_root / "courses").mkdir(parents=True, exist_ok=True)
    resp = client.request("DELETE", "/api/library/nope-999?confirm=nope-999")
    assert resp.status_code == 404
    assert resp.json()["error"] == "course_not_found"


def test_delete_escaping_slug_refused(client, demo_course, libv2_root):
    # A path with traversal slashes never matches the {course_id} route (405);
    # a single-segment but invalid slug is refused by the service (422). Either
    # way the real course survives.
    traversal = client.request(
        "DELETE", "/api/library/..%2f..%2fetc?confirm=..%2f..%2fetc"
    )
    assert traversal.status_code in (404, 405, 422, 400)
    # A single-segment slug that reaches the handler but fails the slug regex
    # (uppercase-leading) is rejected by the service with a 422.
    bad = client.request("DELETE", "/api/library/Bad_Slug?confirm=Bad_Slug")
    assert bad.status_code == 422
    assert bad.json()["error"] in ("invalid_slug", "slug_escapes_root")
    assert (libv2_root / "courses" / demo_course).exists()


# --------------------------------------------------------------------------- #
# /api/courses/{id}/manifest
# --------------------------------------------------------------------------- #


def test_manifest_returns_org_tree(client, demo_course):
    resp = client.get(f"/api/courses/{demo_course}/manifest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == demo_course
    assert body["title"] == "Demo Course Title"
    # ROOT is flattened: top level is the Week 1 module.
    assert len(body["items"]) == 1
    week = body["items"][0]
    assert week["title"] == "Week 1"
    assert week["item_type"] == "module"
    pages = week["children"]
    assert [p["title"] for p in pages] == ["Overview", "Self Check"]
    assert all(p["item_type"] == "page" and p["item"] and p["href"] for p in pages)


def test_manifest_unknown_course_404(client):
    resp = client.get("/api/courses/nope-999/manifest")
    assert resp.status_code == 404
    assert resp.json()["error"] == "course_not_found"


def test_manifest_invalid_slug_422(client):
    # An uppercase-leading / dot-dot slug fails the slug regex before fs touch.
    resp = client.get("/api/courses/..%2f..%2fetc/manifest")
    assert resp.status_code in (404, 422)


# --------------------------------------------------------------------------- #
# /api/courses/{id}/page
# --------------------------------------------------------------------------- #


def test_page_serves_sanitized_html(client, demo_course):
    resp = client.get(f"/api/courses/{demo_course}/page", params={"item": "RES_self_check"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Content-Security-Policy" in resp.headers
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    html = resp.text
    # The active content the page shipped must be stripped.
    assert "toggleAnswer" not in html  # <script> removed
    assert "onclick" not in html.lower()
    assert "onload" not in html.lower()
    assert "javascript:" not in html.lower()
    # The inert component markup (shim reanimates this) survives.
    assert 'class="reveal-btn"' in html
    assert 'aria-controls="ans1"' in html
    # Heading id injected for in-page anchoring.
    assert 'id="self-check"' in html


def test_page_missing_item_422(client, demo_course):
    resp = client.get(f"/api/courses/{demo_course}/page", params={"item": ""})
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_item"


def test_page_unknown_item_404(client, demo_course):
    resp = client.get(f"/api/courses/{demo_course}/page", params={"item": "RES_nope"})
    assert resp.status_code == 404
    assert resp.json()["error"] == "item_not_found"


def test_page_unknown_course_404(client):
    resp = client.get("/api/courses/nope-999/page", params={"item": "RES_overview"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# /api/courses/{id}/asset  (path-traversal + content-type whitelist)
# --------------------------------------------------------------------------- #


def test_asset_serves_whitelisted_png(client, demo_course):
    resp = client.get(f"/api/courses/{demo_course}/asset", params={"path": "assets/diagram.png"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.content.startswith(b"\x89PNG")


def test_asset_serves_whitelisted_css(client, demo_course):
    resp = client.get(f"/api/courses/{demo_course}/asset", params={"path": "assets/style.css"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/css")


@pytest.mark.parametrize(
    "bad_path",
    [
        "../secret.txt",
        "../../etc/passwd",
        "assets/../../secret.txt",
        "/etc/passwd",
        "assets/..%2f..%2fsecret.txt",  # the router decodes the query before us
    ],
)
def test_asset_rejects_traversal(client, demo_course, bad_path):
    resp = client.get(f"/api/courses/{demo_course}/asset", params={"path": bad_path})
    # Either a 422 traversal rejection or a 422 type rejection (no .png/.css ext)
    # — never a 200 that leaks a file outside the cartridge.
    assert resp.status_code in (404, 422)
    assert resp.status_code != 200


def test_asset_rejects_html_as_asset(client, demo_course):
    # HTML pages must route through /page (sanitised), never /asset.
    resp = client.get(
        f"/api/courses/{demo_course}/asset", params={"path": "week_01/overview.html"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "asset_type_not_allowed"


def test_asset_rejects_non_whitelisted_extension(client, demo_course):
    resp = client.get(f"/api/courses/{demo_course}/asset", params={"path": "secret.txt"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "asset_type_not_allowed"


def test_asset_missing_member_404(client, demo_course):
    resp = client.get(f"/api/courses/{demo_course}/asset", params={"path": "assets/nope.png"})
    assert resp.status_code == 404
    assert resp.json()["error"] == "asset_missing"


def test_asset_unknown_course_404(client):
    resp = client.get("/api/courses/nope-999/asset", params={"path": "assets/x.png"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# /api/courses/{id}/source-pdf  (B4 provenance chain — final hop)
# --------------------------------------------------------------------------- #


def _archive_pdf(libv2_root: Path, slug: str, name: str, pages: int = 3) -> Path:
    pdf_dir = libv2_root / "courses" / slug / "source" / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_dir / name
    path.write_bytes(_tiny_pdf(pages))
    return path


def _archive_sidecar(libv2_root: Path, slug: str, doc_slug: str, source_pdf: str) -> None:
    """Archive a {stem}_synthesized.json sidecar recording source_pdf."""
    src_dir = libv2_root / "courses" / slug / "source" / "html"
    src_dir.mkdir(parents=True, exist_ok=True)
    sidecar = src_dir / f"{doc_slug}_synthesized.json"
    sidecar.write_text(
        json.dumps({"slug": doc_slug, "title": doc_slug, "source_pdf": source_pdf}),
        encoding="utf-8",
    )


@pytest.fixture
def pdf_course(libv2_root: Path):
    """A synthetic course archiving one source PDF (single-PDF corpus)."""
    _build_cartridge(libv2_root, "pdf-101")
    _archive_pdf(libv2_root, "pdf-101", "textbook.pdf", pages=3)
    return "pdf-101"


def test_source_pdf_serves_application_pdf(client, pdf_course):
    resp = client.get(
        f"/api/courses/{pdf_course}/source-pdf", params={"file": "textbook", "page": 2}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF-")
    # Single-page extract when a PDF lib is installed, else whole-file fallback.
    assert resp.headers.get("x-source-pdf-mode") in {"single-page", "whole-file"}
    assert resp.headers.get("x-source-pdf-page") == "2"


def test_source_pdf_single_pdf_fallback_when_hint_unmatched(client, pdf_course):
    # An unmatched (benign) file hint on a single-PDF course serves the lone PDF.
    resp = client.get(
        f"/api/courses/{pdf_course}/source-pdf",
        params={"file": "does_not_match", "page": 1},
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-source-pdf-warning") == "single_pdf_fallback"


def test_source_pdf_resolves_via_sidecar_map(client, libv2_root):
    # Multi-PDF course: the sidecar maps a sourceId slug to the right PDF.
    _build_cartridge(libv2_root, "multi-202")
    _archive_pdf(libv2_root, "multi-202", "alpha_textbook.pdf")
    _archive_pdf(libv2_root, "multi-202", "beta_workbook.pdf")
    # sourceId slug "beta-chapter" → beta_workbook.pdf
    _archive_sidecar(libv2_root, "multi-202", "beta-chapter", "/orig/beta_workbook.pdf")
    resp = client.get(
        "/api/courses/multi-202/source-pdf",
        params={"file": "beta-chapter", "page": 1},
    )
    assert resp.status_code == 200
    assert resp.content.startswith(b"%PDF-")
    assert "beta_workbook.pdf" in resp.headers.get("content-disposition", "")


def test_source_pdf_multi_pdf_unresolved_404(client, libv2_root):
    # Multi-PDF course, hint matches nothing, no sidecar → never guess.
    _build_cartridge(libv2_root, "multi-303")
    _archive_pdf(libv2_root, "multi-303", "a.pdf")
    _archive_pdf(libv2_root, "multi-303", "b.pdf")
    resp = client.get(
        "/api/courses/multi-303/source-pdf", params={"file": "zzz", "page": 1}
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "source_pdf_unresolved"


def test_source_pdf_no_archived_pdf_404(client, demo_course):
    # demo-101 has a cartridge but NO source/pdf dir.
    resp = client.get(
        f"/api/courses/{demo_course}/source-pdf", params={"file": "x", "page": 1}
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "no_source_pdf"


@pytest.mark.parametrize("bad_file", ["../secret", "a/b", "x\\y", "..\\x"])
def test_source_pdf_rejects_traversal_file(client, pdf_course, bad_file):
    resp = client.get(
        f"/api/courses/{pdf_course}/source-pdf",
        params={"file": bad_file, "page": 1},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_file"


def test_source_pdf_rejects_bad_page(client, pdf_course):
    # Negative pages are invalid (page=0 now means whole-file mode — see below).
    resp = client.get(
        f"/api/courses/{pdf_course}/source-pdf", params={"file": "textbook", "page": -1}
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_page"


def test_source_pdf_whole_file_mode_page_zero(client, pdf_course):
    """page=0 serves the WHOLE archived PDF (the browse-surface affordance)."""
    resp = client.get(
        f"/api/courses/{pdf_course}/source-pdf", params={"file": "textbook", "page": 0}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.headers["x-source-pdf-mode"] == "whole-file"
    # The whole file (all pages) is served, not a single extracted page.
    assert resp.content.startswith(b"%PDF")


def test_source_pdf_unknown_course_404(client):
    resp = client.get("/api/courses/nope-999/source-pdf", params={"file": "x", "page": 1})
    assert resp.status_code == 404
    assert resp.json()["error"] == "course_not_found"


def test_source_pdf_invalid_slug_422(client):
    resp = client.get(
        "/api/courses/..%2f..%2fetc/source-pdf", params={"file": "x", "page": 1}
    )
    assert resp.status_code in (404, 422)


@pytest.mark.skipif(
    not _has_pdf_lib(), reason="no pypdf / PyMuPDF — single-page extract unavailable"
)
def test_source_pdf_single_page_extract(client, pdf_course):
    resp = client.get(
        f"/api/courses/{pdf_course}/source-pdf", params={"file": "textbook", "page": 2}
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-source-pdf-mode") == "single-page"
    from gui.services import source_pdf as svc  # noqa: PLC0415

    extracted = svc._extract_single_page(_tiny_pdf(3), 2)
    assert extracted is not None and extracted.startswith(b"%PDF-")


@pytest.mark.skipif(
    not _has_pdf_lib(), reason="no pypdf / PyMuPDF — page-range check unavailable"
)
def test_source_pdf_page_out_of_range_404(client, pdf_course):
    resp = client.get(
        f"/api/courses/{pdf_course}/source-pdf", params={"file": "textbook", "page": 99}
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "page_out_of_range"

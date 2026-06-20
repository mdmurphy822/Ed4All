"""FastAPI ``TestClient`` tests for the source-materials router + service.

Covers the three new endpoints (``/api/courses/{id}/source-materials``,
``/source-doc``, ``/source-doc-asset``) over SYNTHETIC tmp-path course archives —
a builder writes tiny DART HTML docs + figure assets + PDFs into
``<libv2_root>/courses/<slug>/...`` (no real LibV2 course is touched; state
isolates via the ``libv2_root`` fixture). Synthetic slugs only.

Asserts the happy path AND the security contract: the active-content sanitiser
(script/onclick/javascript: stripped), block-anchor injection (``id="dart-…"``
where missing, existing ids never clobbered), the three-form ``?ref=`` anchor
shapes, figure-src rewrite, the full path-traversal negative matrix on ``doc``
and ``path``, the exact CSP/nosniff headers, and the operator toggle (env off +
per-course manifest override).

Skipped on a default install with no ``fastapi`` / ``bs4`` (the ``gui`` extra is
opt-in).
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

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "pipeline"
if str(_FIXTURE_DIR) not in sys.path:
    sys.path.insert(0, str(_FIXTURE_DIR))
from build_fixture_pdf import build_pdf  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic archive builder
# --------------------------------------------------------------------------- #

# A DART-shaped accessible HTML doc with: a block carrying data-dart-block-id and
# NO id (block-anchor injection target), a block that ALREADY has an id (must not
# be clobbered), a heading (heading-id injection), a relative figure src (rewrite
# target), and hostile active content (script / onload / onclick / javascript:).
_DART_HTML = """<!DOCTYPE html>
<html><head><title>Chapter One — Foundations</title>
<script>function boom(){alert('xss');}</script>
</head>
<body onload="boom()">
  <h1>Chapter One</h1>
  <h2>Real Numbers</h2>
  <ul class="dart-section" data-dart-block-role="list_unordered"
      data-dart-block-id="blk_target" data-dart-pages="4">
    <li>An ordered list block (no id — gets id="dart-blk_target").</li>
  </ul>
  <p id="sec-existing-h" data-dart-block-id="blk_keep">A block with an existing id.</p>
  <button onclick="boom()">Reveal</button>
  <a href="javascript:alert(2)">bad link</a>
  <img src="chapter-one_figures/diagram.png" alt="A diagram">
</body></html>
"""


def _build_dart_course(
    libv2_root: Path,
    slug: str,
    *,
    root: str = "source/html",
    doc: str = "chapter-one",
    with_figures: bool = True,
    with_pdf: bool = True,
    manifest_viewer: dict | None = None,
) -> str:
    """Write a synthetic DART-archive course; return the slug."""
    course = libv2_root / "courses" / slug
    html_dir = course / root
    html_dir.mkdir(parents=True, exist_ok=True)
    (html_dir / f"{doc}.html").write_text(_DART_HTML, encoding="utf-8")
    if with_figures:
        fig_dir = html_dir / f"{doc}_figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        (fig_dir / "diagram.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    if with_pdf:
        pdf_dir = course / "source" / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        (pdf_dir / f"{doc}.pdf").write_bytes(build_pdf([["Page 1."], ["Page 2."]]))
    manifest: dict = {"slug": slug, "features": {"source_provenance": True}}
    if manifest_viewer is not None:
        manifest["viewer"] = manifest_viewer
    (course / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return slug


@pytest.fixture
def client(state_dir, libv2_root):
    """A TestClient over a fresh FULL-mode app with isolated state + LibV2."""
    return TestClient(create_app())


@pytest.fixture
def dart_course(libv2_root: Path) -> str:
    return _build_dart_course(libv2_root, "alpha-101")


# --------------------------------------------------------------------------- #
# /source-materials inventory
# --------------------------------------------------------------------------- #


def test_inventory_happy_path(client, dart_course):
    resp = client.get(f"/api/courses/{dart_course}/source-materials")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == dart_course
    assert body["enabled"] is True
    docs = body["dart_docs"]
    assert len(docs) == 1
    d = docs[0]
    assert d["doc"] == "chapter-one"
    assert d["title"] == "Chapter One — Foundations"  # parsed <title>
    assert d["root"] == "source/html"
    assert d["has_figures"] is True
    assert d["size"] > 0
    pdfs = body["pdfs"]
    assert len(pdfs) == 1
    assert pdfs[0]["name"] == "chapter-one.pdf"


def test_inventory_alternate_root_discovery(client, libv2_root):
    # A course whose DART HTML lives under the legacy sources/textbooks/ root.
    slug = _build_dart_course(
        libv2_root, "beta-202", root="sources/textbooks", with_pdf=False
    )
    resp = client.get(f"/api/courses/{slug}/source-materials")
    assert resp.status_code == 200
    body = resp.json()
    assert [d["doc"] for d in body["dart_docs"]] == ["chapter-one"]
    assert body["dart_docs"][0]["root"] == "sources/textbooks"
    assert body["pdfs"] == []


def test_inventory_first_root_wins_per_stem(client, libv2_root):
    # Same stem under two roots — sources/textbooks wins (first in _DART_ROOTS).
    slug = _build_dart_course(libv2_root, "gamma-303", root="sources/textbooks")
    _build_dart_course(libv2_root, "gamma-303", root="source/html")  # second root
    resp = client.get(f"/api/courses/{slug}/source-materials")
    docs = resp.json()["dart_docs"]
    assert len(docs) == 1
    assert docs[0]["root"] == "sources/textbooks"


def test_inventory_empty_for_chunks_only_course(client, libv2_root):
    # A course with neither DART HTML nor PDFs (chunks-only) → empty inventory.
    course = libv2_root / "courses" / "empty-404"
    (course / "imscc_chunks").mkdir(parents=True)
    resp = client.get("/api/courses/empty-404/source-materials")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["dart_docs"] == []
    assert body["pdfs"] == []


def test_inventory_unknown_course_404(client):
    resp = client.get("/api/courses/nope-999/source-materials")
    assert resp.status_code == 404
    assert resp.json()["error"] == "course_not_found"


@pytest.mark.parametrize("bad", ["../etc", "WithCaps../x", "a/b"])
def test_inventory_invalid_slug_422(client, bad):
    resp = client.get(f"/api/courses/{bad}/source-materials")
    assert resp.status_code in (404, 422)  # routing may 404 on slashes
    if resp.status_code == 422:
        assert resp.json()["error"] == "invalid_course_id"


# --------------------------------------------------------------------------- #
# /source-doc serving + sanitisation + anchor injection
# --------------------------------------------------------------------------- #


def test_source_doc_serves_sanitized_html(client, dart_course):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc", params={"doc": "chapter-one"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    html = resp.text
    # Active content stripped.
    assert "<script" not in html
    assert "onload=" not in html
    assert "onclick=" not in html
    assert "javascript:alert" not in html
    # Original content preserved.
    assert "Real Numbers" in html


def test_source_doc_injects_block_anchor(client, dart_course):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc", params={"doc": "chapter-one"}
    )
    html = resp.text
    # The block with data-dart-block-id and no id gets id="dart-{block}".
    assert 'id="dart-blk_target"' in html


def test_source_doc_never_clobbers_existing_id(client, dart_course):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc", params={"doc": "chapter-one"}
    )
    html = resp.text
    # The block that already carried id="sec-existing-h" keeps it (never
    # clobbered) — but the #dart-<block> fragment contract must still hold:
    # an empty inline anchor span with the dart id is planted inside it.
    assert 'id="sec-existing-h"' in html
    assert '<span class="dart-block-anchor" id="dart-blk_keep"></span>' in html


def test_source_doc_injects_page_anchor(client, dart_course):
    """Phase 3: the block carrying data-dart-pages="4" yields a #page-4 anchor.

    ``blk_target`` already received id="dart-blk_target" from the block-anchor
    pass, so the page anchor is planted as an inline span (never clobbers).
    """
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc", params={"doc": "chapter-one"}
    )
    html = resp.text
    assert 'id="page-4"' in html
    assert '<span class="dart-page-anchor" id="page-4"></span>' in html


def test_source_doc_accepts_page_query_param(client, dart_course):
    """A ?page=4 request still serves 200 with the #page-4 anchor present.

    The anchor is injected unconditionally, so #page-4 resolves whether or not
    ?page rides along — the route just absorbs the param without 422-ing.
    """
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc",
        params={"doc": "chapter-one", "page": "4"},
    )
    assert resp.status_code == 200
    assert 'id="page-4"' in resp.text


def test_source_doc_absent_page_byte_identical(client, dart_course):
    """RISK-B: serving with vs without ?page= produces byte-identical HTML."""
    base = client.get(
        f"/api/courses/{dart_course}/source-doc", params={"doc": "chapter-one"}
    ).text
    with_page = client.get(
        f"/api/courses/{dart_course}/source-doc",
        params={"doc": "chapter-one", "page": "4"},
    ).text
    assert base == with_page


def test_source_doc_rewrites_figure_src(client, dart_course):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc", params={"doc": "chapter-one"}
    )
    html = resp.text
    # The relative {stem}_figures/ src is rewritten to the asset endpoint.
    assert "chapter-one_figures/diagram.png" not in html
    # bs4 serializes & as &amp; in attribute values (correct HTML).
    assert (
        "/api/courses/alpha-101/source-doc-asset?doc=chapter-one&amp;path=diagram.png"
        in html
    )


def test_source_doc_csp_headers_exact(client, dart_course):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc", params={"doc": "chapter-one"}
    )
    assert resp.headers["content-security-policy"] == (
        "default-src 'none'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"
    )
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["referrer-policy"] == "no-referrer"


@pytest.mark.parametrize(
    "ref", ["blk_target", "sec-existing-h", "Real Numbers"]
)
def test_source_doc_ref_three_forms_all_serve(client, dart_course, ref):
    # All three anchor forms (block-id, element id, sec-+normalized heading) serve
    # the doc 200 — anchor resolution is client-side via the #fragment, so a miss
    # degrades to top-of-document with the doc still served (never a hard fail).
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc",
        params={"doc": "chapter-one", "ref": ref},
    )
    assert resp.status_code == 200


def test_source_doc_unknown_stem_404(client, dart_course):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc", params={"doc": "does-not-exist"}
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "source_doc_missing"


@pytest.mark.parametrize(
    "bad_doc",
    ["../secret", "a/b", "x\\y", "..", "chapter-one/../chapter-one"],
)
def test_source_doc_rejects_traversal_doc(client, dart_course, bad_doc):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc", params={"doc": bad_doc}
    )
    # 422 (pre-rejected as a hostile doc) — never a path read.
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_doc"


def test_source_doc_missing_doc_param_422(client, dart_course):
    resp = client.get(f"/api/courses/{dart_course}/source-doc")
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_doc"


# --------------------------------------------------------------------------- #
# /source-doc-asset serving + path safety
# --------------------------------------------------------------------------- #


def test_source_doc_asset_serves_whitelisted_image(client, dart_course):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc-asset",
        params={"doc": "chapter-one", "path": "diagram.png"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")
    # Asset CSP posture (defence in depth even on whitelisted types).
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_source_doc_asset_rejects_html(client, dart_course):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc-asset",
        params={"doc": "chapter-one", "path": "diagram.html"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "asset_type_not_allowed"


@pytest.mark.parametrize(
    "bad_path",
    ["../secret.png", "a/../../x.png", "x\\y.png", "/etc/passwd.png"],
)
def test_source_doc_asset_rejects_traversal(client, dart_course, bad_path):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc-asset",
        params={"doc": "chapter-one", "path": bad_path},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_asset_path"


def test_source_doc_asset_nul_rejected(client, dart_course):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc-asset",
        params={"doc": "chapter-one", "path": "x\x00.png"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_asset_path"


def test_source_doc_asset_unknown_member_404(client, dart_course):
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc-asset",
        params={"doc": "chapter-one", "path": "missing.png"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "asset_missing"


def test_source_doc_asset_symlink_escape_rejected(client, libv2_root, dart_course):
    # A symlink inside the figures dir pointing OUTSIDE the course must be
    # commonpath-rejected (catches a symlink the pre-rejection can't see).
    course = libv2_root / "courses" / dart_course
    fig_dir = course / "source" / "html" / "chapter-one_figures"
    outside = libv2_root / "secret.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n" + b"secret")
    link = fig_dir / "escape.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc-asset",
        params={"doc": "chapter-one", "path": "escape.png"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_asset_path"


# --------------------------------------------------------------------------- #
# Operator toggle (§2.5)
# --------------------------------------------------------------------------- #


def test_toggle_env_off_disables_inventory(client, dart_course, monkeypatch):
    monkeypatch.setenv("ED4ALL_SOURCE_MATERIALS", "off")
    resp = client.get(f"/api/courses/{dart_course}/source-materials")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["dart_docs"] == []
    assert body["pdfs"] == []


def test_toggle_env_off_403s_source_doc(client, dart_course, monkeypatch):
    monkeypatch.setenv("ED4ALL_SOURCE_MATERIALS", "off")
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc", params={"doc": "chapter-one"}
    )
    assert resp.status_code == 403
    assert resp.json()["error"] == "source_materials_disabled"


def test_toggle_env_off_403s_source_doc_asset(client, dart_course, monkeypatch):
    monkeypatch.setenv("ED4ALL_SOURCE_MATERIALS", "off")
    resp = client.get(
        f"/api/courses/{dart_course}/source-doc-asset",
        params={"doc": "chapter-one", "path": "diagram.png"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"] == "source_materials_disabled"


def test_toggle_env_off_403s_source_pdf(client, dart_course, monkeypatch):
    # /source-pdf is folded under the same toggle (a deliberate behavior change).
    monkeypatch.setenv("ED4ALL_SOURCE_MATERIALS", "off")
    resp = client.get(
        f"/api/courses/{dart_course}/source-pdf",
        params={"file": "chapter-one", "page": 1},
    )
    assert resp.status_code == 403
    assert resp.json()["error"] == "source_materials_disabled"


def test_per_course_override_beats_env_off(client, libv2_root, monkeypatch):
    # Env off, but the per-course manifest explicitly exposes source materials.
    slug = _build_dart_course(
        libv2_root, "delta-505", manifest_viewer={"expose_source_materials": True}
    )
    monkeypatch.setenv("ED4ALL_SOURCE_MATERIALS", "off")
    resp = client.get(f"/api/courses/{slug}/source-materials")
    body = resp.json()
    assert body["enabled"] is True
    assert len(body["dart_docs"]) == 1


def test_per_course_override_beats_env_on(client, libv2_root, monkeypatch):
    # Env on (default), but the per-course manifest disables this one course.
    slug = _build_dart_course(
        libv2_root, "epsilon-606", manifest_viewer={"expose_source_materials": False}
    )
    monkeypatch.delenv("ED4ALL_SOURCE_MATERIALS", raising=False)
    resp = client.get(f"/api/courses/{slug}/source-materials")
    assert resp.json()["enabled"] is False
    doc_resp = client.get(
        f"/api/courses/{slug}/source-doc", params={"doc": "chapter-one"}
    )
    assert doc_resp.status_code == 403


# --------------------------------------------------------------------------- #
# _DART_ROOTS sync with the canonical citation_anchor definition
# --------------------------------------------------------------------------- #


def test_dart_roots_match_citation_anchor():
    from gui.services import source_materials as sm
    from lib.retrieval import citation_anchor

    assert sm._DART_ROOTS == citation_anchor._DART_ROOTS

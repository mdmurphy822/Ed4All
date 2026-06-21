"""Tests for the source-page viewer service (Executor B / WS4 wave 1).

Covers: the path-traversal attack matrix (pre-resolution rejection +
post-resolution commonpath containment for symlink escapes), in-memory imscc
member serving, untrusted-HTML sanitization, heading-id injection pinned to the
``grounded_answer._fragment_for`` slug, and the CSP / nosniff response headers.

Fixtures: ``tests/fixtures/retrieval/mini_course/`` (WS1, read-only) is copied
into a tmp ``libv2_root/courses/<slug>/`` so the resolver's search roots
(``source/html``, ``source/imscc``) line up.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from gui.services import source_page
from gui.services.source_page import (
    SourcePageError,
    SourcePageResult,
    heading_slug,
    render_source_page,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MINI_COURSE = REPO_ROOT / "tests" / "fixtures" / "retrieval" / "mini_course"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_course(libv2_root: Path, slug: str, *, kind: str) -> Path:
    """Copy the mini-course source tree into ``libv2_root/courses/<slug>``.

    ``kind`` controls which chunks dir is created so
    ``_infer_chunkset_kind`` returns the kind under test (``dart`` ->
    ``dart_chunks/``, ``imscc`` -> ``imscc_chunks/``).
    """
    course_dir = libv2_root / "courses" / slug
    course_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(MINI_COURSE / "source", course_dir / "source")
    chunks_dir = {"dart": "dart_chunks", "imscc": "imscc_chunks", "corpus": "corpus"}[kind]
    cdir = course_dir / chunks_dir
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    return course_dir


@pytest.fixture
def dart_course(libv2_root: Path) -> str:
    slug = "mini-dart"
    _make_course(libv2_root, slug, kind="dart")
    return slug


@pytest.fixture
def imscc_course(libv2_root: Path) -> str:
    slug = "mini-imscc"
    _make_course(libv2_root, slug, kind="imscc")
    return slug


# --------------------------------------------------------------------------- #
# Slug-algorithm pin (R6: must equal _fragment_for's regex)
# --------------------------------------------------------------------------- #


def test_heading_slug_matches_fragment_for():
    """The viewer's id slugger must equal grounded_answer._fragment_for's."""
    import re

    samples = [
        "Retrieval Quality",
        "Vector Stores and Embeddings",
        "Week 1: Application & Activities",
        "  Trailing & Leading  ",
        "ALL CAPS Heading",
        "numbers 123 and symbols!!!",
    ]
    for heading in samples:
        expected = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
        assert heading_slug(heading) == expected, heading


# --------------------------------------------------------------------------- #
# Happy paths — DART file + imscc member
# --------------------------------------------------------------------------- #


def test_dart_page_resolves(dart_course, libv2_root):
    res = render_source_page(dart_course, "alpha.html", libv2_root)
    assert isinstance(res, SourcePageResult)
    assert "Vector Stores and Embeddings" in res.html
    assert res.media_type.startswith("text/html")


def test_imscc_member_resolves_in_memory(imscc_course, libv2_root):
    """imscc member served from the zip with NO on-disk extraction."""
    res = render_source_page(imscc_course, "alpha.html", libv2_root)
    assert "Vector Stores and Embeddings" in res.html
    # No directory was unpacked next to the cartridge.
    course_dir = libv2_root / "courses" / imscc_course
    unpacked = list((course_dir / "source" / "imscc").glob("*_unpacked"))
    assert unpacked == [], "imscc must be read in-memory, not extracted to disk"


# --------------------------------------------------------------------------- #
# Traversal attack matrix — pre-resolution rejection (no fs touch)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_path",
    [
        "../../../../etc/passwd",
        "../alpha.html",
        "a/../../b.html",
        "/etc/passwd",
        "/absolute/page.html",
        "\\\\server\\share\\x.html",
        "dir\\file.html",
        "C:\\Windows\\win.ini",
        "C:/Windows/win.ini",
        "page\x00.html",
        "",
    ],
)
def test_traversal_rejected_pre_resolution(dart_course, libv2_root, bad_path, monkeypatch):
    """Hostile item_paths are rejected 422 BEFORE the resolver runs.

    A sentinel patched over ``resolve_source_page`` proves no fs/zip access
    happened — the guard fires first.
    """
    called = {"hit": False}

    def _sentinel(*a, **k):  # pragma: no cover - must never run
        called["hit"] = True
        raise AssertionError("resolver reached with a hostile item_path")

    monkeypatch.setattr(
        "lib.retrieval.citation_anchor.resolve_source_page", _sentinel
    )

    with pytest.raises(SourcePageError) as ei:
        render_source_page(dart_course, bad_path, libv2_root)
    assert ei.value.status == 422
    assert ei.value.code == "invalid_item_path"
    assert called["hit"] is False


def test_traversal_rejection_is_chdir_independent(dart_course, libv2_root, tmp_path, monkeypatch):
    """The pre-rejection does not depend on the process cwd."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SourcePageError) as ei:
        render_source_page(dart_course, "../../../../etc/passwd", libv2_root)
    assert ei.value.status == 422


# --------------------------------------------------------------------------- #
# Post-resolution containment — symlink escape (legal-looking path)
# --------------------------------------------------------------------------- #


def test_symlink_escape_rejected_by_containment(dart_course, libv2_root, tmp_path):
    """A clean-looking item_path that resolves (via symlink/rglob) outside the
    course dir is rejected by the commonpath containment check."""
    # Plant a secret OUTSIDE any course dir.
    secret = tmp_path / "outside_secret.html"
    secret.write_text("<html><body><h1>TOP SECRET</h1></body></html>", encoding="utf-8")

    course_dir = libv2_root / "courses" / dart_course
    html_root = course_dir / "source" / "html"
    link = html_root / "leak.html"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")

    # "leak.html" is clean (no .., no abs) so it passes pre-rejection; the
    # resolver follows the symlink; containment must reject the escape.
    with pytest.raises(SourcePageError) as ei:
        render_source_page(dart_course, "leak.html", libv2_root)
    assert ei.value.status == 422
    assert ei.value.code == "invalid_item_path"


def test_legal_path_inside_course_serves(dart_course, libv2_root):
    """A non-trivial-but-legal nested path under the source root still serves
    (containment must not reject in-bounds paths)."""
    course_dir = libv2_root / "courses" / dart_course
    nested = course_dir / "source" / "html" / "ch1"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "page.html").write_text(
        "<html><body><h1>Nested Page</h1></body></html>", encoding="utf-8"
    )
    res = render_source_page(dart_course, "ch1/page.html", libv2_root)
    assert "Nested Page" in res.html


# --------------------------------------------------------------------------- #
# Slug shape / course existence
# --------------------------------------------------------------------------- #


def test_unknown_course_404(libv2_root):
    with pytest.raises(SourcePageError) as ei:
        render_source_page("no-such-course", "alpha.html", libv2_root)
    assert ei.value.status == 404
    assert ei.value.code == "course_not_found"


@pytest.mark.parametrize("bad_slug", ["../etc", "..", "/abs", "Has Space", ".hidden"])
def test_malformed_slug_422(libv2_root, bad_slug):
    with pytest.raises(SourcePageError) as ei:
        render_source_page(bad_slug, "alpha.html", libv2_root)
    assert ei.value.status == 422


def test_missing_page_404(dart_course, libv2_root):
    with pytest.raises(SourcePageError) as ei:
        render_source_page(dart_course, "does-not-exist.html", libv2_root)
    assert ei.value.status == 404
    assert ei.value.code == "source_page_missing"


# --------------------------------------------------------------------------- #
# Sanitization — untrusted cartridge HTML
# --------------------------------------------------------------------------- #


HOSTILE_HTML = (
    "<!DOCTYPE html><html><head>"
    "<script>alert('xss')</script>"
    "<link rel='preload' href='evil.js' as='script'>"
    "</head><body>"
    "<h1 onclick=\"steal()\">Hostile Heading</h1>"
    "<p onmouseover='leak()'>Body text</p>"
    "<a href=\"javascript:alert(1)\">bad link</a>"
    "<a href=\"/learn/safe\">good link</a>"
    "<iframe src=\"http://evil.example\"></iframe>"
    "<img src=\"data:image/png;base64,AAAA\" alt=\"ok\">"
    "</body></html>"
)


def _serve_hostile(libv2_root: Path, slug: str) -> str:
    course_dir = _make_course(libv2_root, slug, kind="dart")
    (course_dir / "source" / "html" / "hostile.html").write_text(
        HOSTILE_HTML, encoding="utf-8"
    )
    res = render_source_page(slug, "hostile.html", libv2_root)
    return res.html


def test_script_tag_stripped(libv2_root):
    html = _serve_hostile(libv2_root, "hostile-1")
    assert "<script" not in html.lower()
    assert "alert('xss')" not in html


def test_event_handlers_stripped(libv2_root):
    html = _serve_hostile(libv2_root, "hostile-2")
    assert "onclick" not in html.lower()
    assert "onmouseover" not in html.lower()
    assert "steal()" not in html
    assert "leak()" not in html


def test_javascript_url_stripped(libv2_root):
    html = _serve_hostile(libv2_root, "hostile-3")
    assert "javascript:" not in html.lower()
    # The benign link survives.
    assert "/learn/safe" in html


def test_iframe_and_preload_stripped(libv2_root):
    html = _serve_hostile(libv2_root, "hostile-4")
    assert "<iframe" not in html.lower()
    assert "evil.example" not in html
    assert "<link" not in html.lower()


def test_safe_image_data_uri_preserved(libv2_root):
    """A data: img (allowed by the CSP) and its alt text survive sanitization."""
    html = _serve_hostile(libv2_root, "hostile-5")
    assert 'alt="ok"' in html or "alt='ok'" in html


# --------------------------------------------------------------------------- #
# Heading-id injection
# --------------------------------------------------------------------------- #


def test_heading_ids_injected(dart_course, libv2_root):
    """Headings without ids get the _fragment_for slug; #fragment lands."""
    res = render_source_page(dart_course, "alpha.html", libv2_root)
    # alpha.html has "<h2>Retrieval Quality</h2>"
    assert 'id="retrieval-quality"' in res.html
    # h1 too.
    assert 'id="vector-stores-and-embeddings"' in res.html


def test_existing_heading_ids_preserved(libv2_root):
    course_dir = _make_course(libv2_root, "ids-stable", kind="dart")
    (course_dir / "source" / "html" / "ided.html").write_text(
        '<html><body><h1 id="keep-me">Title</h1>'
        "<h2>Auto Slug Me</h2></body></html>",
        encoding="utf-8",
    )
    res = render_source_page("ids-stable", "ided.html", libv2_root)
    assert 'id="keep-me"' in res.html
    assert "auto-slug-me" in res.html


def test_heading_text_slug_added_when_hash_id_present(libv2_root):
    # DART stamps hash ids (id="sec-<hash>") on ~every heading. The Ask answer
    # deep-link fragment is the SLUGGED heading text (grounded_answer.
    # _fragment_for), so the text-slug must ALSO resolve to the heading — planted
    # as a sibling anchor span — even though the heading already has a hash id.
    # Without this the "view in textbook" link lands at document TOP.
    course_dir = _make_course(libv2_root, "hashids-1", kind="dart")
    (course_dir / "source" / "html" / "hashed.html").write_text(
        '<html><body>'
        '<h2 id="sec-2570ef19b386cdac">Prime Factorization</h2>'
        "</body></html>",
        encoding="utf-8",
    )
    res = render_source_page("hashids-1", "hashed.html", libv2_root)
    # Original hash id preserved.
    assert 'id="sec-2570ef19b386cdac"' in res.html
    # Text-slug planted as a sibling anchor so the heading-slug fragment resolves.
    assert '<span class="dart-heading-anchor" id="prime-factorization"></span>' in res.html


# --------------------------------------------------------------------------- #
# Banner / structure
# --------------------------------------------------------------------------- #


def test_banner_has_no_heading_element(dart_course, libv2_root):
    """The banner is nav+p only — the archived page keeps the unique h1."""
    from bs4 import BeautifulSoup

    res = render_source_page(dart_course, "alpha.html", libv2_root)
    soup = BeautifulSoup(res.html, "html.parser")
    banner = soup.find("nav", attrs={"aria-label": "Source context"})
    assert banner is not None
    assert banner.find(["h1", "h2", "h3", "h4", "h5", "h6"]) is None
    assert banner.find("a") is not None  # "Back to your question"
    # The archived page still has exactly one h1.
    assert len(soup.find_all("h1")) == 1


def test_fragment_note_when_no_heading_fragment(dart_course, libv2_root):
    """fragment=None → banner adds the 'cited section is on this page' note."""
    res = render_source_page(dart_course, "alpha.html", libv2_root, fragment=None)
    assert "The cited section is on this page." in res.html


def test_no_fragment_note_when_heading_fragment(dart_course, libv2_root):
    res = render_source_page(
        dart_course, "alpha.html", libv2_root, fragment="retrieval-quality"
    )
    assert "The cited section is on this page." not in res.html


# --------------------------------------------------------------------------- #
# Response headers (CSP + nosniff)
# --------------------------------------------------------------------------- #


def test_csp_and_nosniff_headers_present(dart_course, libv2_root):
    res = render_source_page(dart_course, "alpha.html", libv2_root)
    assert "Content-Security-Policy" in res.headers
    csp = res.headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp
    assert "img-src 'self' data:" in csp
    assert res.headers["X-Content-Type-Options"] == "nosniff"

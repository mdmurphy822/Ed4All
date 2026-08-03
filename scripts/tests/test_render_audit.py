"""GAP 3 — tests for scripts/harness/render_audit.py.

Pure-function coverage (no browser): the delimiter scanner, duplicate-id
finder, and the assess_page findings mapper. Plus one live smoke test
over a tiny fixture page, skipped when Chromium is unavailable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts" / "harness"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import render_audit as ra  # noqa: E402

# ------------------------------------------------------------------ #
# scan_literal_delimiters
# ------------------------------------------------------------------ #


def test_scan_delimiters_clean() -> None:
    assert ra.scan_literal_delimiters("The current I flows through R.") == {}


def test_scan_delimiters_detects_each_kind() -> None:
    text = r"Un-typeset \( x \) and display \[ y \] and $$z$$ leaked."
    counts = ra.scan_literal_delimiters(text)
    assert counts.get("\\(") == 1
    assert counts.get("\\)") == 1
    assert counts.get("\\[") == 1
    assert counts.get("\\]") == 1
    assert counts.get("$$") == 2  # opening + closing


def test_scan_delimiters_empty_text() -> None:
    assert ra.scan_literal_delimiters("") == {}


# ------------------------------------------------------------------ #
# find_duplicate_ids
# ------------------------------------------------------------------ #


def test_duplicate_ids_none() -> None:
    assert ra.find_duplicate_ids(["a", "b", "c"]) == {}


def test_duplicate_ids_found() -> None:
    dupes = ra.find_duplicate_ids(["a", "b", "a", "c", "a", "b"])
    assert dupes == {"a": 3, "b": 2}


def test_duplicate_ids_ignores_empty() -> None:
    assert ra.find_duplicate_ids(["", "", "x"]) == {}


# ------------------------------------------------------------------ #
# assess_page
# ------------------------------------------------------------------ #


def _clean_raw() -> dict:
    return {
        "ids": ["a", "b", "c"],
        "mjx_merror_count": 0,
        "main_count": 1,
        "skip_link_count": 1,
        "img_missing_alt": 0,
        "text_outside_mjx": "All good, math typeset fine.",
    }


def test_assess_clean_page_passes() -> None:
    audit = ra.assess_page("clean", "/x.html", _clean_raw())
    assert audit.failed is False
    assert audit.findings == []


def test_assess_flags_merror() -> None:
    raw = _clean_raw()
    raw["mjx_merror_count"] = 2
    audit = ra.assess_page("e", "/x.html", raw)
    assert audit.failed is True
    assert any(f.code == "MJX_MERROR" for f in audit.findings)


def test_assess_flags_literal_delimiters() -> None:
    raw = _clean_raw()
    raw["text_outside_mjx"] = r"leaked \( x \)"
    audit = ra.assess_page("d", "/x.html", raw)
    assert audit.failed is True
    assert any(f.code == "LITERAL_DELIMITERS" for f in audit.findings)


def test_assess_flags_duplicate_ids() -> None:
    raw = _clean_raw()
    raw["ids"] = ["a", "a", "b"]
    audit = ra.assess_page("dup", "/x.html", raw)
    assert audit.failed is True
    assert any(f.code == "DUPLICATE_IDS" for f in audit.findings)


def test_assess_flags_missing_and_extra_main() -> None:
    for count in (0, 2):
        raw = _clean_raw()
        raw["main_count"] = count
        audit = ra.assess_page("m", "/x.html", raw)
        assert any(f.code == "MISSING_MAIN" for f in audit.findings)
        assert audit.failed is True


def test_assess_flags_missing_skip_link() -> None:
    raw = _clean_raw()
    raw["skip_link_count"] = 0
    audit = ra.assess_page("s", "/x.html", raw)
    assert any(f.code == "MISSING_SKIP_LINK" for f in audit.findings)


def test_assess_missing_alt_is_warning_not_failure() -> None:
    raw = _clean_raw()
    raw["img_missing_alt"] = 3
    audit = ra.assess_page("alt", "/x.html", raw)
    assert any(
        f.code == "IMG_MISSING_ALT" and f.severity == "warning"
        for f in audit.findings
    )
    # Warning alone must not fail the page.
    assert audit.failed is False


def test_assess_notes_mathjax_timeout() -> None:
    raw = _clean_raw()
    raw["mathjax_timeout"] = True
    audit = ra.assess_page("t", "/x.html", raw)
    assert any("timed out" in n for n in audit.notes)


def test_to_dict_shape() -> None:
    audit = ra.assess_page("clean", "/x.html", _clean_raw())
    d = audit.to_dict()
    for key in ("name", "path", "failed", "findings", "notes", "counts"):
        assert key in d
    assert d["counts"]["main"] == 1


# ------------------------------------------------------------------ #
# Live fixture smoke (skipped when Chromium unavailable).
# ------------------------------------------------------------------ #


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            b.close()
        return True
    except Exception:
        return False


_FIXTURE_CLEAN = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Clean</title></head><body>
<a href="#main" class="skip-link">Skip to content</a>
<main id="main"><h1>Hello</h1><p>The resistance is R and current I.</p>
<img src="x.png" alt="a diagram"></main></body></html>"""

_FIXTURE_DIRTY = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Dirty</title></head><body>
<div id="dup"></div><div id="dup"></div>
<p>Leaked math \\( x + y \\) here.</p>
<img src="y.png"></body></html>"""


@pytest.mark.skipif(not _chromium_available(),
                    reason="Chromium not available for live render test")
def test_live_render_audit_fixture(tmp_path: Path) -> None:
    clean = tmp_path / "clean.html"
    clean.write_text(_FIXTURE_CLEAN, encoding="utf-8")
    dirty = tmp_path / "dirty.html"
    dirty.write_text(_FIXTURE_DIRTY, encoding="utf-8")

    audits = ra.inspect_pages(
        [("clean", str(clean)), ("dirty", str(dirty))]
    )
    by_name = {a.name: a for a in audits}

    assert by_name["clean"].failed is False, by_name["clean"].to_dict()

    dirty_audit = by_name["dirty"]
    assert dirty_audit.failed is True
    codes = {f.code for f in dirty_audit.findings}
    assert "DUPLICATE_IDS" in codes
    assert "LITERAL_DELIMITERS" in codes
    assert "MISSING_MAIN" in codes
    assert "MISSING_SKIP_LINK" in codes
    assert "IMG_MISSING_ALT" in codes

"""Regression: Defect 1 — document-level repeated page-furniture strip.

The VLM emits a per-page running footer / header as its own markdown line at
the page top / bottom; the P1 fusion then glues it into body prose. These tests
exercise the deterministic cross-page detector + strip in isolation (no VLM
endpoint, no OCR, no model). Synthetic multi-page line lists only — never a
course-data path.
"""

from __future__ import annotations

import pytest

from semantik_structure import vlm_furniture as vf


def test_detect_masks_page_numbers_in_running_furniture():
    """A header/footer differing only by page number shares one signature."""
    pages = [
        ["Chapter 9 Roots and Radicals", "content one", "This book is free at http://x.org/col/803"],
        ["Chapter 9 Roots and Radicals", "content two", "This book is free at http://x.org/col/815"],
        ["Chapter 9 Roots and Radicals", "content three", "This book is free at http://x.org/col/827"],
        ["9.1 A Unique Real Section", "wholly distinct body text here"],
    ]
    sigs = vf.detect_furniture_signatures(pages)
    # Both the header and the number-varying footer are detected.
    assert vf.normalize_furniture_sig("Chapter 9 Roots and Radicals") in sigs
    assert (
        vf.normalize_furniture_sig("This book is free at http://x.org/col/803")
        in sigs
    )
    # The genuine section title recurs on ONE page → never furniture.
    assert vf.normalize_furniture_sig("9.1 A Unique Real Section") not in sigs


def test_normalize_masks_digits_and_markers():
    a = vf.normalize_furniture_sig("Chapter 9 Roots and Radicals 803")
    b = vf.normalize_furniture_sig("# Chapter 9 Roots and Radicals 815")
    assert a == b  # digit runs masked, leading markdown marker stripped
    assert "#" in a  # the masked digit placeholder survives


def test_strip_removes_matching_lines_anywhere():
    pages = [["Foo Bar Header", "body", "Foo Bar Header"]] * 3
    sigs = vf.detect_furniture_signatures(pages)
    stripped = vf.strip_furniture_lines(pages[0], sigs)
    assert "body" in stripped
    assert "Foo Bar Header" not in stripped


def test_short_signatures_never_furniture():
    """A bare section number '9.1' must never be masked into furniture."""
    pages = [["9.1", "unique a"], ["9.2", "unique b"], ["9.3", "unique c"]]
    sigs = vf.detect_furniture_signatures(pages)
    assert sigs == set()  # '#.#' is below the min-signature length


def test_below_min_pages_returns_empty():
    pages = [["Header", "x"], ["Header", "y"]]  # 2 pages < _MIN_PAGES
    assert vf.detect_furniture_signatures(pages) == set()


def test_min_share_threshold_tunable():
    # Header on 2 of 5 pages = 0.4 share.
    pages = [
        ["Running Header Line", "a"],
        ["Running Header Line", "b"],
        ["Distinct Title One", "c"],
        ["Distinct Title Two", "d"],
        ["Distinct Title Three", "e"],
    ]
    hdr = vf.normalize_furniture_sig("Running Header Line")
    assert hdr not in vf.detect_furniture_signatures(pages, min_share=0.5)
    assert hdr in vf.detect_furniture_signatures(pages, min_share=0.3)


def test_resolve_strip_mode_default_on(monkeypatch):
    monkeypatch.delenv("SEMANTIK_VLM_STRIP_FURNITURE", raising=False)
    assert vf.resolve_strip_furniture_mode() is True
    monkeypatch.setenv("SEMANTIK_VLM_STRIP_FURNITURE", "0")
    assert vf.resolve_strip_furniture_mode() is False
    monkeypatch.setenv("SEMANTIK_VLM_STRIP_FURNITURE", "garbage")
    assert vf.resolve_strip_furniture_mode() is True


def test_resolve_min_share_parse_with_fallback(monkeypatch):
    monkeypatch.setenv("SEMANTIK_VLM_FURNITURE_MIN_SHARE", "0.6")
    assert vf.resolve_furniture_min_share() == pytest.approx(0.6)
    for bad in ("", "nan", "-1", "2", "abc"):
        monkeypatch.setenv("SEMANTIK_VLM_FURNITURE_MIN_SHARE", bad)
        assert vf.resolve_furniture_min_share() == pytest.approx(0.35)


# --------------------------------------------------------------------------
# Wiring: extract_shared._strip_document_furniture over synthetic page dicts.
# --------------------------------------------------------------------------


def _page(lines):
    return {
        "vlm": {"text_blocks": [{"text": t} for t in lines]},
        "tesseract": {"text_blocks": [{"text": t} for t in lines]},
    }


def test_strip_document_furniture_wiring(monkeypatch):
    from semantik_structure import extract_shared

    monkeypatch.delenv("SEMANTIK_VLM_STRIP_FURNITURE", raising=False)
    pages = [
        _page(["Chapter 9 Roots and Radicals", "unique body one", "Free book at http://x.org/col/803"]),
        _page(["Chapter 9 Roots and Radicals", "unique body two", "Free book at http://x.org/col/815"]),
        _page(["Chapter 9 Roots and Radicals", "unique body three", "Free book at http://x.org/col/827"]),
    ]
    extract_shared._strip_document_furniture(pages)
    for page in pages:
        vlm_texts = [b["text"] for b in page["vlm"]["text_blocks"]]
        assert "Chapter 9 Roots and Radicals" not in vlm_texts
        assert not any("Free book at" in t for t in vlm_texts)
        assert any("unique body" in t for t in vlm_texts)
        assert page["vlm_furniture_dropped"] == 2  # header + footer per page


def test_strip_document_furniture_respects_disable_flag(monkeypatch):
    from semantik_structure import extract_shared

    monkeypatch.setenv("SEMANTIK_VLM_STRIP_FURNITURE", "0")
    pages = [
        _page(["Running Header X", "a"]),
        _page(["Running Header X", "b"]),
        _page(["Running Header X", "c"]),
    ]
    extract_shared._strip_document_furniture(pages)
    # Disabled → untouched (no furniture key, header survives).
    assert all("Running Header X" in [b["text"] for b in p["vlm"]["text_blocks"]] for p in pages)
    assert all("vlm_furniture_dropped" not in p for p in pages)


# --------------------------------------------------------------------------
# Coordinator follow-up (Defect A): tesseract-side footer + apparatus safety.
# --------------------------------------------------------------------------


def _tess_page(lines):
    """Tesseract-side page: bboxes in top-to-bottom order (y grows downward)."""
    return {
        "tesseract": {
            "text_blocks": [
                {"text": t, "bbox": [10.0, 100.0 * i, 200.0, 100.0 * i + 10.0]}
                for i, t in enumerate(lines)
            ]
        },
        "vlm": {"text_blocks": []},
    }


def test_tesseract_side_repeated_last_block_footer_dropped(monkeypatch):
    """(i) A footer clean in TESSERACT (last block) is detected + dropped even
    when the VLM lines carry no margin-position copy of it."""
    from semantik_structure import extract_shared

    monkeypatch.delenv("SEMANTIK_VLM_STRIP_FURNITURE", raising=False)
    pages = [
        _tess_page(["alpha body", "beta body", "This book is free at http://x.org/col/803"]),
        _tess_page(["gamma body", "delta body", "This book is free at http://x.org/col/815"]),
        _tess_page(["eps body", "zeta body", "This book is free at http://x.org/col/827"]),
    ]
    extract_shared._strip_document_furniture(pages)
    for p in pages:
        texts = [b["text"] for b in p["tesseract"]["text_blocks"]]
        assert not any("free at" in t for t in texts), texts
        assert len(texts) == 2  # only the footer dropped
        assert p["tesseract_furniture_dropped"] == 1


def test_union_signature_strips_vlm_footer_anywhere(monkeypatch):
    """A footer voted in by the TESSERACT side is stripped from the VLM lines
    even when it sits MID-page there (the VLM position-gate gap)."""
    from semantik_structure import extract_shared

    monkeypatch.delenv("SEMANTIK_VLM_STRIP_FURNITURE", raising=False)

    words = ["alpha", "beta", "gamma", "delta"]

    def _page(i):
        w = words[i]
        page = _tess_page(
            [f"body {w} opens", f"body {w} continues", f"Book free at http://x.org/col/{800 + i}"]
        )
        # VLM reads the footer MID-page (position 2 of 6 — outside both margins).
        page["vlm"]["text_blocks"] = [
            {"text": f"body {w} opens"},
            {"text": f"body {w} continues"},
            {"text": f"Book free at http://x.org/col/{800 + i}"},
            {"text": f"exercise {w} first part"},
            {"text": f"exercise {w} second part"},
            {"text": f"exercise {w} third part"},
        ]
        return page

    pages = [_page(i) for i in range(4)]
    extract_shared._strip_document_furniture(pages)
    for p in pages:
        vlm_texts = [b["text"] for b in p["vlm"]["text_blocks"]]
        assert not any("free at" in t for t in vlm_texts), vlm_texts
        assert len(vlm_texts) == 5


def test_once_only_apparatus_line_never_furniture_while_header_stripped(monkeypatch):
    """(ii)+(iv) A once-only all-caps apparatus line at page top is NOT
    classified as furniture, while the true cross-page running header IS."""
    from semantik_structure import extract_shared

    monkeypatch.delenv("SEMANTIK_VLM_STRIP_FURNITURE", raising=False)
    header = "Chapter 3 Sample Topic"

    def _page(first_line, word):
        return _tess_page(
            [header, first_line, f"body {word} paragraph", f"closing {word} sentence"]
        )

    pages = [
        _page("PRACTICE TEST", "alpha"),  # apparatus: appears ONCE, at page top
        _page("section prose a", "beta"),
        _page("section prose b", "gamma"),
        _page("section prose c", "delta"),
    ]
    extract_shared._strip_document_furniture(pages)
    all_texts = [
        b["text"] for p in pages for b in p["tesseract"]["text_blocks"]
    ]
    assert "PRACTICE TEST" in all_texts  # once-only apparatus survives
    assert header not in all_texts  # the repeated running header is stripped


def test_short_label_at_margin_never_furniture():
    """A recurring 1-2-token pedagogical label at a margin is NOT furniture
    (the _MIN_SIG_TOKENS anti-FP guard for worked-example-dense pages)."""
    import semantik_structure.vlm_furniture as vf

    pages = [["Solution", "unique alpha body", "unique alpha close"],
             ["Solution", "unique beta body", "unique beta close"],
             ["Solution", "unique gamma body", "unique gamma close"]]
    assert vf.detect_furniture_signatures(pages) == set()

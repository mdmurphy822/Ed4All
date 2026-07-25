"""``--auto-name`` H1-derived, run-timestamped course slugs — pure helpers.

Locks ``lib/course_autoname.py``:

* slug composition (canonical_slug + run-init timestamp, whole-token cap),
* bounded h1 extraction from accessible HTML,
* every fallback arm of the honest fallback matrix (never fabricate a title),
* run-init timestamp resolution (created_at ISO > run_id suffix > None).

All fixtures are synthetic — no course-data references (project rule).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from lib.course_autoname import (
    AUTO_SLUG_TITLE_MAX_CHARS,
    compose_auto_slug,
    extract_h1_title,
    resolve_auto_course_name,
    resolve_run_init_timestamp,
    title_rejection_reason,
    truncate_slug_whole_tokens,
)

RUN_INIT = datetime(2026, 7, 22, 7, 4, 33)


# ---------------------------------------------------------------------------
# Slug composition
# ---------------------------------------------------------------------------


def test_compose_auto_slug_title_case_and_timestamp():
    assert (
        compose_auto_slug("Principles Of Sample Systems", RUN_INIT)
        == "principles-of-sample-systems-20260722-0704"
    )


def test_compose_auto_slug_punctuation_via_canonical_slug():
    # canonical_slug DELETES disallowed chars (digits fuse: "2.2" -> "22").
    assert (
        compose_auto_slug("Systems, Design & Practice: 2.2!", RUN_INIT)
        == "systems-design-practice-22-20260722-0704"
    )


def test_compose_auto_slug_caps_title_at_whole_token_boundary():
    title = " ".join(["alpha"] * 30)  # slug would be 179 chars untruncated
    slug = compose_auto_slug(title, RUN_INIT)
    base = slug.rsplit("-20260722-0704", 1)[0]
    assert len(base) <= AUTO_SLUG_TITLE_MAX_CHARS
    # Whole-token truncation: no token is split mid-word.
    assert set(base.split("-")) == {"alpha"}
    assert slug.endswith("-20260722-0704")


def test_truncate_slug_whole_tokens_keeps_oversized_first_token():
    long_token = "x" * 80
    assert truncate_slug_whole_tokens(long_token, 60) == long_token


def test_truncate_slug_short_input_unchanged():
    assert truncate_slug_whole_tokens("a-b-c", 60) == "a-b-c"


# ---------------------------------------------------------------------------
# H1 extraction
# ---------------------------------------------------------------------------


def _write_html(tmp_path, body, name="doc_accessible.html"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_extract_h1_title_strips_nested_tags_and_entities(tmp_path):
    p = _write_html(
        tmp_path,
        '<html><body><h1 id="t"><span>Signals &amp; Systems</span></h1>'
        "<h1>Second Heading Ignored</h1></body></html>",
    )
    assert extract_h1_title(p) == "Signals & Systems"


def test_extract_h1_title_collapses_whitespace(tmp_path):
    p = _write_html(tmp_path, "<h1>\n  A   Multi\n  Line\tTitle </h1>")
    assert extract_h1_title(p) == "A Multi Line Title"


def test_extract_h1_title_none_when_absent(tmp_path):
    p = _write_html(tmp_path, "<html><h2>Only a subheading</h2></html>")
    assert extract_h1_title(p) is None


def test_extract_h1_title_none_on_missing_file(tmp_path):
    assert extract_h1_title(tmp_path / "nope.html") is None


def test_extract_h1_title_bounded_read(tmp_path):
    # h1 sits BEYOND the read bound -> honestly not found (never a full slurp).
    p = _write_html(tmp_path, ("x" * 2048) + "<h1>Late Title</h1>")
    assert extract_h1_title(p, max_bytes=1024) is None


# ---------------------------------------------------------------------------
# Fallback matrix — title_rejection_reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title,reason",
    [
        (None, "h1_missing"),
        ("", "h1_missing"),
        ("   ", "h1_missing"),
        ("Chapter 3", "h1_structural"),
        ("chapter 12:", "h1_structural"),
        ("Part IV", "h1_structural"),
        ("Unit 2 -", "h1_structural"),
        ("Appendix A", "h1_structural"),
        ("1.2.3", "h1_numeric"),
        ("42", "h1_numeric"),
        ("!!! ???", "slug_empty"),
        ("y" * 121, "h1_too_long"),
    ],
)
def test_title_rejection_matrix(title, reason):
    assert title_rejection_reason(title) == reason


@pytest.mark.parametrize(
    "title",
    [
        "Principles Of Sample Systems",
        "Chapter Books for Beginners",  # 'chapter' + non-ordinal = real title
        "Part-Time Systems Engineering",
        "y" * 120,  # exactly at the ceiling
    ],
)
def test_title_acceptance(title):
    assert title_rejection_reason(title) is None


# ---------------------------------------------------------------------------
# Run-init timestamp resolution
# ---------------------------------------------------------------------------


def test_resolve_run_init_prefers_created_at():
    ts = resolve_run_init_timestamp("2026-07-22T07:04:33.123456", "TTC_x_20250101_010101")  # slug-guard: allow
    assert ts == datetime(2026, 7, 22, 7, 4, 33, 123456)


def test_resolve_run_init_falls_back_to_run_id():
    ts = resolve_run_init_timestamp("not-a-date", "TTC_prov_20260722_070433")  # slug-guard: allow
    assert ts == datetime(2026, 7, 22, 7, 4, 33)


def test_resolve_run_init_none_when_unresolvable():
    assert resolve_run_init_timestamp(None, "WF-20260722-abcd1234") is None
    assert resolve_run_init_timestamp("", None) is None


# ---------------------------------------------------------------------------
# resolve_auto_course_name — end-to-end pure resolution
# ---------------------------------------------------------------------------


def test_resolve_happy_path(tmp_path):
    p = _write_html(tmp_path, "<h1>Principles Of Sample Systems</h1>")
    res = resolve_auto_course_name("prov-name", [str(p)], RUN_INIT)
    assert res.resolved is True
    assert res.reason == "h1_resolved"
    assert res.final_name == "principles-of-sample-systems-20260722-0704"
    assert res.display_title == "Principles Of Sample Systems"


def test_resolve_multi_file_corpus_keeps_provisional(tmp_path):
    a = _write_html(tmp_path, "<h1>Title A</h1>", "a_accessible.html")
    b = _write_html(tmp_path, "<h1>Title B</h1>", "b_accessible.html")
    res = resolve_auto_course_name("prov-name", [str(a), str(b)], RUN_INIT)
    assert res.resolved is False
    assert res.reason == "multi_file_corpus"
    assert res.final_name == "prov-name"


def test_resolve_no_conversion_output_keeps_provisional():
    res = resolve_auto_course_name("prov-name", [], RUN_INIT)
    assert res.resolved is False
    assert res.reason == "no_conversion_output"
    assert res.final_name == "prov-name"


def test_resolve_no_run_timestamp_keeps_provisional(tmp_path):
    p = _write_html(tmp_path, "<h1>Real Title</h1>")
    res = resolve_auto_course_name("prov-name", [str(p)], None)
    assert res.resolved is False
    assert res.reason == "no_run_timestamp"
    assert res.final_name == "prov-name"


def test_resolve_garbage_h1_keeps_provisional(tmp_path):
    p = _write_html(tmp_path, "<h1>Chapter 1</h1>")
    res = resolve_auto_course_name("prov-name", [str(p)], RUN_INIT)
    assert res.resolved is False
    assert res.reason == "h1_structural"
    assert res.final_name == "prov-name"
    assert res.h1_title == "Chapter 1"


def test_resolve_missing_h1_keeps_provisional(tmp_path):
    p = _write_html(tmp_path, "<p>No headings at all</p>")
    res = resolve_auto_course_name("prov-name", [str(p)], RUN_INIT)
    assert res.resolved is False
    assert res.reason == "h1_missing"
    assert res.final_name == "prov-name"

"""Package 1 + 3 — SemantiK structure-fidelity extractor guards.

Exercises the ``ED4ALL_STRUCTURE_EXTRACT_GUARDS`` article-path assembly guards
on SYNTHETIC ``<article role="doc-chapter">`` HTML (no course-data path, no
model). Covers:

- 1a continuation-article merge (a headless second article folds into the
  previous chapter instead of minting ``Chapter 2``).
- 1b headingless ``<section class="dart-section">`` wrappers group as
  contentBlocks under the preceding heading-bearing section.
- 1c a noncontent / numbered-apparatus heading ("Preface", "1.4 Exercises")
  does not mint a section — its content regroups.
- 1d structureDiagnostics counters + the >3x overcount WARNING.
- flag-off byte-identical snapshot (default OFF).
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.semantic_structure_extractor import SemanticStructureExtractor  # noqa: E402


_GUARDS_ENV = "ED4ALL_STRUCTURE_EXTRACT_GUARDS"
_ANCHOR_ENV = "ED4ALL_STRUCTURE_OUTLINE_ANCHOR"


@pytest.fixture(autouse=True)
def _anchor_off(monkeypatch):
    """Pin Package-2 outline anchoring OFF for the whole Package-1 suite.

    Package 2 (``ED4ALL_STRUCTURE_OUTLINE_ANCHOR``) defaults ON when the
    Package-1 guards are on; these tests exercise the Package-1 assembly in
    isolation, so the satellite is switched off. ``guards on + anchor off`` is
    exactly the Package-1 behavior these snapshots pin.
    """
    monkeypatch.setenv(_ANCHOR_ENV, "0")


def _extract(html: str):
    return SemanticStructureExtractor().extract(html)


# ---------------------------------------------------------------------------
# Fixtures — DPUB-ARIA doc-chapter articles with flat per-block wrappers.
# ---------------------------------------------------------------------------


# Two articles: the second has NO visible content h1/h2 (continuation). The
# first carries a real section, two headingless wrappers, and a numbered
# "1.4 Exercises" apparatus banner.
_TWO_ARTICLE_HTML = """
<html lang="en"><body><main>
  <nav class="toc"><p>1.1 Whole Numbers</p><p>1.2 Integers</p></nav>
  <article role="doc-chapter" id="chap-1">
    <header><h2 id="c1">Foundations</h2></header>
    <section class="dart-section"><h3>1.1 Whole Numbers</h3><p>Whole numbers intro.</p></section>
    <section class="dart-section"><p>Stray paragraph one, no heading.</p></section>
    <section class="dart-section"><p>Stray paragraph two, no heading.</p></section>
    <section class="dart-section"><h3>1.4 Exercises</h3><p>Do problems 1-20.</p></section>
  </article>
  <article role="doc-chapter" id="chap-2">
    <section class="dart-section"><h3>1.2 Integers</h3><p>Integers continue here.</p></section>
    <section class="dart-section"><p>Continuation body block.</p></section>
  </article>
</main></body></html>
"""


# ---------------------------------------------------------------------------
# 1a — continuation-article merge.
# ---------------------------------------------------------------------------


def test_continuation_article_merges_into_previous(monkeypatch):
    monkeypatch.setenv(_GUARDS_ENV, "1")
    res = _extract(_TWO_ARTICLE_HTML)
    # One chapter, not two — the headless second article merged in. No
    # "Chapter 2" fallback title is minted.
    assert len(res["chapters"]) == 1
    assert res["chapters"][0]["headingText"] == "Foundations"
    titles = [c["headingText"] for c in res["chapters"]]
    assert "Chapter 2" not in titles
    diag = res["structureDiagnostics"]["guards"]
    assert diag["continuations_merged"] == 1


def test_first_article_never_merges(monkeypatch):
    # A single headless first article still mints a chapter (nothing precedes).
    monkeypatch.setenv(_GUARDS_ENV, "1")
    html = (
        '<html lang="en"><body><main>'
        '<article role="doc-chapter" id="chap-1">'
        '<section class="dart-section"><p>Body with no chapter heading.</p></section>'
        "</article></main></body></html>"
    )
    res = _extract(html)
    assert len(res["chapters"]) == 1
    assert res["structureDiagnostics"]["guards"]["continuations_merged"] == 0


# ---------------------------------------------------------------------------
# 1b — headingless wrapper grouping.
# ---------------------------------------------------------------------------


def test_headingless_wrappers_group_under_preceding_section(monkeypatch):
    monkeypatch.setenv(_GUARDS_ENV, "1")
    wrappers = "".join(
        f'<section class="dart-section"><p>Body block {i}.</p></section>'
        for i in range(5)
    )
    html = (
        '<html lang="en"><body><main>'
        '<article role="doc-chapter" id="chap-1">'
        '<header><h2>Foundations</h2></header>'
        '<section class="dart-section"><h3>1.1 Whole Numbers</h3><p>Lead prose.</p></section>'
        f"{wrappers}"
        "</article></main></body></html>"
    )
    res = _extract(html)
    chapter = res["chapters"][0]
    # Exactly ONE section (the heading-bearing one); the 5 headingless
    # wrappers grouped into it as contentBlocks (nothing dropped).
    assert len(chapter["sections"]) == 1
    sec = chapter["sections"][0]
    assert sec["headingText"] == "1.1 Whole Numbers"
    # 1 heading block + 1 lead <p> + 5 grouped <p> = 7 content blocks.
    assert len(sec["contentBlocks"]) == 7
    assert res["structureDiagnostics"]["guards"]["headingless_wrappers_grouped"] == 5
    # No empty-heading sections survive.
    assert all(s["headingText"] for s in chapter["sections"])


def test_headingless_wrappers_before_any_section_go_to_lead(monkeypatch):
    monkeypatch.setenv(_GUARDS_ENV, "1")
    html = (
        '<html lang="en"><body><main>'
        '<article role="doc-chapter" id="chap-1">'
        '<header><h2>Foundations</h2></header>'
        '<section class="dart-section"><p>Orphan lead block.</p></section>'
        '<section class="dart-section"><h3>1.1 Whole Numbers</h3><p>Real section.</p></section>'
        "</article></main></body></html>"
    )
    res = _extract(html)
    chapter = res["chapters"][0]
    # The orphan headingless block (no preceding section) grouped into the
    # chapter's implicit lead contentBlocks.
    lead_texts = " ".join(str(b) for b in chapter["contentBlocks"])
    assert "Orphan lead block" in json.dumps(chapter["contentBlocks"])
    assert len(chapter["sections"]) == 1


# ---------------------------------------------------------------------------
# 1c — noncontent / numbered-apparatus heading does not mint a section.
# ---------------------------------------------------------------------------


def test_numbered_exercises_banner_does_not_mint_section(monkeypatch):
    monkeypatch.setenv(_GUARDS_ENV, "1")
    res = _extract(_TWO_ARTICLE_HTML)
    all_section_titles = [
        s["headingText"]
        for c in res["chapters"]
        for s in c["sections"]
    ]
    assert "1.4 Exercises" not in all_section_titles
    # The exercises content is preserved (regrouped), not dropped.
    assert "Do problems 1-20." in json.dumps(res["chapters"])
    assert res["structureDiagnostics"]["guards"]["apparatus_demoted"] >= 1


def test_noncontent_heading_filtered_on_article_path(monkeypatch):
    monkeypatch.setenv(_GUARDS_ENV, "1")
    html = (
        '<html lang="en"><body><main>'
        '<article role="doc-chapter" id="chap-1">'
        '<header><h2>Foundations</h2></header>'
        '<section class="dart-section"><h3>Preface</h3><p>Front matter prose.</p></section>'
        '<section class="dart-section"><h3>1.1 Whole Numbers</h3><p>Real prose.</p></section>'
        "</article></main></body></html>"
    )
    res = _extract(html)
    titles = [s["headingText"] for s in res["chapters"][0]["sections"]]
    assert "Preface" not in titles
    assert "1.1 Whole Numbers" in titles
    # Front-matter content preserved (regrouped into the chapter lead).
    assert "Front matter prose." in json.dumps(res["chapters"][0])


# ---------------------------------------------------------------------------
# 1d — diagnostics + overcount warning.
# ---------------------------------------------------------------------------


def test_diagnostics_fields_populated(monkeypatch):
    monkeypatch.setenv(_GUARDS_ENV, "1")
    diag = _extract(_TWO_ARTICLE_HTML)["structureDiagnostics"]["guards"]
    assert set(diag) == {
        "sections_built",
        "headingless_wrappers_grouped",
        "continuations_merged",
        "apparatus_demoted",
        "declared_section_estimate",
    }
    assert diag["sections_built"] == 2
    assert diag["headingless_wrappers_grouped"] == 3
    assert diag["continuations_merged"] == 1
    assert diag["apparatus_demoted"] == 1
    # nav.toc declares 1.1 + 1.2 = 2 distinct N.M ordinals.
    assert diag["declared_section_estimate"] == 2


def test_overcount_warning_fires(monkeypatch, caplog):
    monkeypatch.setenv(_GUARDS_ENV, "1")
    secs = "".join(
        f'<section class="dart-section"><h3>1.{i} Topic {i}</h3><p>x</p></section>'
        for i in range(1, 10)
    )
    html = (
        '<html lang="en"><body><main>'
        '<nav class="toc"><p>1.1 Only One Declared</p></nav>'
        '<article role="doc-chapter" id="c1"><header><h2>Foundations</h2></header>'
        f"{secs}</article></main></body></html>"
    )
    with caplog.at_level(logging.WARNING):
        res = _extract(html)
    # 9 built vs 1 declared → >3x → warning.
    assert res["structureDiagnostics"]["guards"]["sections_built"] == 9
    assert "STRUCTURE_SECTION_OVERCOUNT" in caplog.text


def test_no_overcount_warning_within_ratio(monkeypatch, caplog):
    monkeypatch.setenv(_GUARDS_ENV, "1")
    with caplog.at_level(logging.WARNING):
        _extract(_TWO_ARTICLE_HTML)
    assert "STRUCTURE_SECTION_OVERCOUNT" not in caplog.text


# ---------------------------------------------------------------------------
# Flag-off byte-identical (default OFF).
# ---------------------------------------------------------------------------


# The legacy (pre-guard) article-path output for _TWO_ARTICLE_HTML: two
# chapters (the second gets the "Chapter 2" fallback title), and the flat
# per-block wrappers become sections — including two empty-heading sections
# and a "1.4 Exercises" section. Pinned as the flag-off snapshot.
_LEGACY_STRUCTURE = [
    (
        "Foundations",
        ["1.1 Whole Numbers", "", "", "1.4 Exercises"],
    ),
    (
        "Chapter 2",
        ["1.2 Integers", ""],
    ),
]


def _structure_shape(res):
    return [
        (c["headingText"], [s["headingText"] for s in c["sections"]])
        for c in res["chapters"]
    ]


def test_flag_off_is_default_and_legacy(monkeypatch):
    monkeypatch.delenv(_GUARDS_ENV, raising=False)
    res = _extract(_TWO_ARTICLE_HTML)
    assert _structure_shape(res) == _LEGACY_STRUCTURE
    # No guard diagnostics key when off.
    assert "guards" not in res.get("structureDiagnostics", {})


def _stable_payload(res):
    # Drop the volatile documentInfo (carries an extractionTimestamp) — the
    # snapshot pins the STRUCTURAL surface (chapters + diagnostics).
    payload = copy.deepcopy(res)
    payload.pop("documentInfo", None)
    return json.dumps(payload, sort_keys=True)


def test_flag_off_byte_identical_snapshot(monkeypatch):
    # Two independent extractions with the flag off serialize identically
    # (determinism, timestamp aside) AND match the pinned legacy structure —
    # the guard code is fully inert when the flag is unset.
    monkeypatch.delenv(_GUARDS_ENV, raising=False)
    a = _stable_payload(_extract(_TWO_ARTICLE_HTML))
    b = _stable_payload(_extract(copy.deepcopy(_TWO_ARTICLE_HTML)))
    assert a == b
    assert _structure_shape(json.loads(a)) == _LEGACY_STRUCTURE


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_flag_truthy_values_enable(monkeypatch, val):
    monkeypatch.setenv(_GUARDS_ENV, val)
    res = _extract(_TWO_ARTICLE_HTML)
    assert len(res["chapters"]) == 1  # guard active

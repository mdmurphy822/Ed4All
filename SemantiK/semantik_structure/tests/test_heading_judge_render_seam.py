"""END-TO-END seam: a judged heading LEVEL must reach the rendered ``<hN>`` tag.

The pre-existing coverage tested the two halves separately — ``apply_judged_levels``
re-stamping ``level`` on the provenance (``test_heading_judge.py``) and the render
mapping a provenance ``level`` onto an ``<hN>`` tag
(``lib/semantik/tests/test_cascade_ir_chapter_ladder.py``) — so both suites stayed
green while the SEAM between them was never exercised. This module drives the
WHOLE chain on one fixture:

    layout sidecar
      -> transform_document        (pending headings default to level 3)
      -> judge verdict             (stub seat; no GPU, no network)
      -> apply_judged_levels       (clamp rules re-stamp ``level``)
      -> corrected_layout.json     (--apply)
      -> render_accessible_html    (build_chapters_ir -> adapter)
      -> the rendered <hN> tag

and asserts the OBSERVABLE difference: an unjudged pending heading renders
``<h4>``; the same heading judged L3->L2 renders ``<h3>``. That is the assertion
that was missing all along.

It also pins the complementary property (2026-07-22 investigation): when the
verdict CONFIRMS every existing level, the judged re-render is byte-identical to
the unjudged render. A byte-identical judged output is therefore the CORRECT
outcome of an agreeing (or already-applied) judge pass, not evidence of a broken
seam — the distinction the live incident turned on.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from semantik_structure.glmocr import heading_judge_standalone as hjs
from semantik_structure.glmocr.transform import GlmPage, transform_document

# The render half of the seam lives in Ed4All's ``lib/semantik`` (the adapter).
# A pure-SemantiK venv has no ``lib/`` — skip rather than fail there.
pytest.importorskip("lib.semantik.adapter")


# ── Fixture: a two-chapter book with pending sub-chapter headings. ───────────
# Region ids are the flat reading-order index (== ``first_raw_block_index``),
# which is what a judge verdict is keyed by.
_PAGE_1: List[Tuple[str, str]] = [
    ("doc_title", "# Sample Widget Compendium"),      # 0
    ("paragraph_title", "Chapter 1"),                  # 1  chapter opener (L1)
    ("paragraph_title", "Introduction"),               # 2  PENDING (L3)
    ("text", "This chapter introduces widgets and the vocabulary used later."),
    ("paragraph_title", "1.1 Alpha Widgets"),          # 4  N.M spine (L2)
    ("text", "Alpha widgets behave predictably under load and are easy to model."),
    ("paragraph_title", "Delta Widgets"),              # 6  PENDING (L3), in 1.1
    ("text", "Delta widgets are a specialised sub-family of alpha widgets."),
    ("image", ""),                                     # 8  figure
    ("figure_title", "Figure 1: An alpha widget assembly."),
    ("paragraph_title", "1.2 Beta Widgets"),           # 10 N.M spine (L2)
    ("text", "Beta widgets differ from alpha widgets in three important ways."),
]
_PAGE_2: List[Tuple[str, str]] = [
    ("paragraph_title", "Chapter 2"),                  # 12 chapter opener (L1)
    ("paragraph_title", "Reliable Widgets"),           # 13 PENDING (L3)
    ("text", "Reliability is the property that a widget keeps working over time."),
    ("paragraph_title", "2.1 Failure Modes"),          # 15 N.M spine (L2)
    ("text", "A widget fails in one of three well understood ways."),
]

#: Heading texts of the three pending headings (see the fixture comments).
#: ``Introduction`` / ``Reliable Widgets`` hang off a chapter opener, so L2 is
#: legal for them; ``Delta Widgets`` is nested inside the 1.1 N.M section, so
#: the no-orphaning clamp forbids it rising to the spine's own level.
_PENDING_TEXTS = ("Introduction", "Delta Widgets", "Reliable Widgets")


def _pages() -> List[GlmPage]:
    pages: List[GlmPage] = []
    idx = 0
    for page_no, items in ((1, _PAGE_1), (2, _PAGE_2)):
        regions: List[Dict[str, Any]] = []
        for native_label, content in items:
            top = 10 + 30 * len(regions)
            regions.append({
                "index": idx,
                "native_label": native_label,
                "bbox_2d": [10, top, 500, top + 30],
                "content": content,
            })
            idx += 1
        pages.append(GlmPage(page_no=page_no, regions=regions))
    return pages


def _pending_ids() -> Dict[str, int]:
    """Map pending heading TEXT -> the region id a judge verdict is keyed by.

    Derived from the transform rather than hardcoded: caption folding means the
    emitted ``first_raw_block_index`` is not the raw layout list position.
    """
    ids = {
        str(p.get("heading_text")): int(p["first_raw_block_index"])
        for p in transform_document(_pages()).region_provenance
        if p.get("region_kind") == "heading" and p.get("heading_level_pending")
    }
    assert set(ids) == set(_PENDING_TEXTS), ids
    return ids


def _write_layout(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "sample.glmocr_layout.json"
    path.write_text(
        json.dumps({
            "schema": "glmocr-layout/1.0",
            "pages": [
                {"page_no": p.page_no, "error": None, "regions": p.regions}
                for p in _pages()
            ],
        }),
        encoding="utf-8",
    )
    return path


class _StubPost:
    """``post_fn(messages, max_tokens) -> (content, finish)`` — no seat, no GPU."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    def __call__(self, messages, max_tokens):  # noqa: ANN001 - test stub
        self.calls += 1
        return self._content, "stop"


def _render_unjudged() -> str:
    """The conversion render with the judge OFF (every pending stays level 3)."""
    from semantik_structure.glmocr.lane import (
        GlmOcrLaneResult,
        render_accessible_html,
    )

    tr = transform_document(_pages())
    result = GlmOcrLaneResult(
        pdf="sample.pdf",
        region_provenance=tr.region_provenance,
        heading_tree=[tuple(t) for t in tr.heading_tree],
        escalations=tr.escalations,
    )
    return render_accessible_html(result, pdf_stem="sample")


def _run_judge(tmp_path: Path, levels: Dict[str, int]) -> Tuple[Dict[str, Any], str]:
    """Run the standalone judge ``--apply`` with a stubbed verdict.

    ``levels`` is keyed by pending heading TEXT (resolved to region ids here).
    Returns ``(report, judged_html)``. ``use_cache=False`` keeps the pass
    hermetic (no content-addressed sidecar reads/writes).
    """
    ids = _pending_ids()
    layout = _write_layout(tmp_path)
    out_dir = tmp_path / "judged"
    body = json.dumps(
        {"levels": {str(ids[text]): lvl for text, lvl in levels.items()}}
    )
    report = hjs.run_standalone(
        layout, out_dir=out_dir, apply=True,
        post_fn=_StubPost(body), use_cache=False,
    )
    html = (out_dir / "sample_accessible.html").read_text(encoding="utf-8")
    return report, html


def _headings(html: str) -> List[Tuple[str, str]]:
    return [
        (m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip())
        for m in re.finditer(r"<h([1-6])[^>]*>(.*?)</h\1>", html, re.S)
    ]


def _level_of(html: str, text: str) -> str | None:
    for level, txt in _headings(html):
        if txt == text:
            return level
    return None


# ── The keystone: a judged level change reaches the rendered tag. ────────────
def test_judged_level_change_changes_the_rendered_heading_tag(tmp_path):
    """L3 pending renders <h4>; the SAME heading judged to L2 renders <h3>.

    This is the end-to-end assertion the split unit suites never made: it fails
    if the level is dropped ANYWHERE on the chain — the transform's pending
    stamp, ``apply_judged_levels``' re-stamp key, the corrected-layout
    round-trip, or the adapter's level -> ``<hN>`` mapping.
    """
    unjudged = _render_unjudged()
    assert _level_of(unjudged, "Introduction") == "4", unjudged
    assert _level_of(unjudged, "Reliable Widgets") == "4", unjudged

    _, judged = _run_judge(
        tmp_path, {"Introduction": 2, "Delta Widgets": 3, "Reliable Widgets": 2}
    )

    assert _level_of(judged, "Introduction") == "3", judged
    assert _level_of(judged, "Reliable Widgets") == "3", judged
    # The un-changed verdict (L3 -> L3) must NOT move its tag.
    assert _level_of(judged, "Delta Widgets") == "4", judged
    # ... and the fixed N.M spine is never re-levelled by the judge.
    assert _level_of(judged, "1.1 Alpha Widgets") == "3", judged
    assert judged != unjudged, (
        "a judged level change must produce a DIFFERENT render; a byte-identical "
        "output here means the judged level never reached the renderer"
    )


def test_corrected_layout_records_the_changed_level(tmp_path):
    """The ``--apply`` sidecar carries the new ``level`` + the judged marker.

    Distinguishes 'a verdict was APPLIED' from 'the level actually CHANGED' —
    the accounting distinction the 2026-07-22 triage turned on.
    """
    report, _ = _run_judge(
        tmp_path, {"Introduction": 2, "Delta Widgets": 3, "Reliable Widgets": 2}
    )
    assert report["n_pending"] == 3
    assert report["applied"] == 3

    ids = _pending_ids()
    corrected = json.loads(
        (tmp_path / "judged" / "sample.corrected_layout.json").read_text(
            encoding="utf-8")
    )
    by_id = {
        p.get("first_raw_block_index"): p
        for p in corrected["region_provenance"]
        if p.get("region_kind") == "heading"
    }
    assert by_id[ids["Introduction"]]["level"] == 2
    assert by_id[ids["Introduction"]]["heading_level_judged"] == {
        "from": 3, "to": 2, "clamped": False,
    }
    # applied-but-unchanged: a recorded verdict that leaves the level alone.
    assert by_id[ids["Delta Widgets"]]["level"] == 3
    assert by_id[ids["Delta Widgets"]]["heading_level_judged"]["from"] == 3
    assert by_id[ids["Delta Widgets"]]["heading_level_judged"]["to"] == 3
    # the pending flag is cleared once a verdict lands.
    assert "heading_level_pending" not in by_id[ids["Introduction"]]

    changed = sum(
        1 for p in by_id.values()
        if (p.get("heading_level_judged") or {}).get("from")
        != (p.get("heading_level_judged") or {}).get("to")
        and p.get("heading_level_judged")
    )
    assert changed == 2, "2 of the 3 applied verdicts actually changed a level"


def test_agreeing_verdict_re_renders_byte_identically(tmp_path):
    """A verdict that CONFIRMS the existing level leaves the render untouched.

    Pins the 2026-07-22 finding: an APPLIED verdict is not a level CHANGE, so a
    byte-identical judged HTML is the CORRECT outcome of an agreeing judge — it
    must never on its own be read as a broken judge->render seam. (The judged
    heading here keeps level 3 and its ``heading_level_pending`` flag is
    cleared, proving the render keys on ``level`` alone.)
    """
    unjudged = _render_unjudged()
    report, judged = _run_judge(tmp_path, {"Delta Widgets": 3})
    assert report["applied"] == 1 and report["kept"] == 2
    assert (
        hashlib.sha256(judged.encode()).hexdigest()
        == hashlib.sha256(unjudged.encode()).hexdigest()
    ), "a level-preserving verdict must not perturb the render"


def test_judge_pass_is_idempotent_on_the_same_layout(tmp_path):
    """Re-judging the same layout with the same verdict reproduces the bytes.

    This is why a `heading_judge` PHASE that re-judges a conversion whose
    in-lane judge already ran (``SEMANTIK_HEADING_JUDGE`` gates BOTH) copies
    back a file identical to its own ``.prejudge.bak`` — an idempotent re-run,
    not a dropped verdict.
    """
    levels = {"Introduction": 2, "Delta Widgets": 3, "Reliable Widgets": 2}
    _, first = _run_judge(tmp_path / "a", levels)
    _, second = _run_judge(tmp_path / "b", levels)
    assert first == second
    assert first != _render_unjudged()


# ── Contracts a re-levelling render must not regress. ────────────────────────
def test_judged_render_preserves_figcaptions_and_ladder_invariants(tmp_path):
    """Exactly one <h1>, one 'Chapter N' series at a single level, captions kept.

    A heading re-level must not mint a phantom chapter, split the chapter
    ladder across two levels, or drop a ``<figcaption>``.
    """
    unjudged = _render_unjudged()
    _, judged = _run_judge(
        tmp_path, {"Introduction": 2, "Delta Widgets": 3, "Reliable Widgets": 2}
    )

    for name, html in (("unjudged", unjudged), ("judged", judged)):
        levels = [lvl for lvl, _ in _headings(html)]
        assert levels.count("1") == 1, (name, _headings(html))
        chapter_levels = {
            lvl for lvl, txt in _headings(html)
            if re.fullmatch(r"Chapter \d+", txt)
        }
        assert chapter_levels == {"2"}, (name, chapter_levels)

    assert judged.count("<figcaption") == unjudged.count("<figcaption") == 1
    assert "An alpha widget assembly." in judged

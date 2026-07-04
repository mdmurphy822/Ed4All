"""Adapter-seam tests — SEMANTIK_OCR_CONFUSABLE_REPAIR render contract.

A ``region_provenance`` entry carrying the additive ``repaired_text`` +
``ocr_repair`` keys must render the REPAIRED text in the body, stamp
``data-dart-repair`` / ``data-dart-repair-count`` on the ``<section>`` OPENING
tag, and carry the repaired text (+ a mirrored ``repair`` object) in the
``_synthesized.json`` sidecar — WITHOUT moving the content-hash sourceId (which
stays pinned to the verbatim ``raw_text``). An entry with no repair keys is
byte-identical to the pre-repair render.

Built with NO models / GPU (synthetic provenance → real adapter path).
"""

from __future__ import annotations

import re

from lib.semantik.adapter import normalize_cascade_to_ed4all
from lib.semantik.cascade_ir import build_chapters_ir


def _prov(*, repaired: bool) -> list[dict]:
    """Two regions: a chapter heading + one content paragraph. The paragraph
    optionally carries the additive OCR-repair keys."""
    para: dict = {
        "region_index": 1,
        "region_kind": "paragraph",
        "role": "paragraph",
        "confidence": 0.95,
        "wcag_status": "passed",
        "first_raw_block_index": 1,
        "pages": [1],
        "heading_text": None,
        "level": None,
        "figure_alt": None,
        # raw_text stays VERBATIM (the garbled OCR text) in both cases.
        "raw_text": "Solve Vx = 4 for the value of x.",
    }
    if repaired:
        para["repaired_text"] = "Solve √x = 4 for the value of x."
        para["ocr_repair"] = {
            "type": "ocr-confusable",
            "n_edits": 1,
            "edits": [
                {"before": "V", "after": "√", "rule_id": "v_to_radical",
                 "token_index": 1},
            ],
        }
    return [
        {
            "region_index": 0,
            "region_kind": "heading",
            "role": "heading",
            "confidence": 0.95,
            "wcag_status": "passed",
            "first_raw_block_index": 0,
            "pages": [1],
            "heading_text": "Chapter 1: Foundations",
            "level": 1,
            "figure_alt": None,
            "raw_text": "Chapter 1: Foundations",
        },
        para,
    ]


def _result(prov: list[dict]) -> dict:
    return {
        "region_provenance": prov,
        "heading_tree": [(1, "Chapter 1: Foundations")],
        "exit_action": "ship_with_flag",
        "wcag_status": "passed",
        "theta_score": 0.8,
        "flags": [],
        "lane_used": "fast",
    }


def _normalize(prov: list[dict]) -> dict:
    chapters = build_chapters_ir(_result(prov))
    return normalize_cascade_to_ed4all(
        {"chapters": chapters, "exit_action": "ship_with_flag",
         "wcag_status": "passed", "theta_score": 0.8, "flags": [],
         "lane_used": "fast"},
        pdf_stem="chapter_01",
    )


# ---------------------------------------------------------------------------
# HTML render.
# ---------------------------------------------------------------------------


def test_repaired_block_renders_repaired_text_and_marker():
    out = _normalize(_prov(repaired=True))
    html = out["html"]
    assert "Solve √x = 4" in html
    assert 'data-dart-repair="ocr-confusable"' in html
    assert 'data-dart-repair-count="1"' in html
    # Placement: the marker is on the same <section ...> opening tag as
    # data-dart-block-id (never a leaf node).
    m = re.search(r"<section [^>]*data-dart-repair=\"ocr-confusable\"[^>]*>", html)
    assert m is not None
    assert "data-dart-block-id=" in m.group(0)


def test_unrepaired_block_is_byte_identical_baseline():
    baseline = _normalize(_prov(repaired=False))["html"]
    assert "data-dart-repair" not in baseline
    assert "Solve Vx = 4" in baseline  # verbatim garbled text (no repair)


# ---------------------------------------------------------------------------
# Content-hash sourceId pinned to ORIGINAL raw_text.
# ---------------------------------------------------------------------------


def test_content_hash_sid_pinned_to_raw_text(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "1")
    repaired_html = _normalize(_prov(repaired=True))["html"]
    plain_html = _normalize(_prov(repaired=False))["html"]

    def _block_ids(html: str) -> set[str]:
        return set(re.findall(r'data-dart-block-id="([^"]+)"', html))

    # raw_text is identical across repair on/off, so the content-hash sids are
    # identical — the repair never moves a sourceId.
    assert _block_ids(repaired_html) == _block_ids(plain_html)


# ---------------------------------------------------------------------------
# Sidecar carries repaired text + repair object; block-id parity.
# ---------------------------------------------------------------------------


def test_sidecar_repaired_text_and_repair_object():
    out = _normalize(_prov(repaired=True))
    sidecar = out["synthesized_sidecar"]
    para = next(
        s for s in sidecar["sections"] if s["section_type"] == "paragraph-group"
    )
    assert para["data"]["text"] == "Solve √x = 4 for the value of x."
    repair = para["data"]["repair"]
    assert repair["type"] == "ocr-confusable"
    assert repair["n_edits"] == 1
    assert repair["original_text"] == "Solve Vx = 4 for the value of x."

    # Block-id parity: every sidecar section_id is a data-dart-block-id in HTML.
    html_ids = set(re.findall(r'data-dart-block-id="([^"]+)"', out["html"]))
    for s in sidecar["sections"]:
        assert s["section_id"] in html_ids


def test_replay_check_reverts_on_tampered_repair():
    # A repaired_text that does NOT replay the recorded edits is discarded
    # (fail-closed): the block renders verbatim raw_text with NO marker.
    prov = _prov(repaired=True)
    prov[1]["repaired_text"] = "Solve √y = 9 tampered."  # not the recorded edit
    out = _normalize(prov)
    html = out["html"]
    assert "data-dart-repair" not in html
    assert "Solve Vx = 4" in html  # reverted to verbatim

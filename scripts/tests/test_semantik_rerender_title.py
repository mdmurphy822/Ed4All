"""Unit tests — semantik_rerender.py --title / --title-map override (fix d).

Exercises the title-resolution helper directly and an end-to-end --from-html
re-render that pins the document <h1>/<title>. CPU-only, no models.
"""

from __future__ import annotations

import json
from pathlib import Path

import importlib.util

_SCRIPT = Path(__file__).resolve().parents[1] / "semantik_rerender.py"
_spec = importlib.util.spec_from_file_location("semantik_rerender", _SCRIPT)
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)


# ---------------------------------------------------------------------------
# Title-resolution helper.
# ---------------------------------------------------------------------------


def test_explicit_title_wins(tmp_path):
    tm = tmp_path / "map.json"
    tm.write_text(json.dumps({"scan-ch04": "From Map"}), encoding="utf-8")
    assert rr._resolve_title_override("scan-ch04", "Explicit", str(tm)) == "Explicit"


def test_title_map_exact_stem(tmp_path):
    tm = tmp_path / "map.json"
    tm.write_text(json.dumps({"sample-algebra-2e-scan-ch04": "Graphs"}), encoding="utf-8")
    got = rr._resolve_title_override("sample-algebra-2e-scan-ch04", None, str(tm))
    assert got == "Graphs"


def test_title_map_chNN_token(tmp_path):
    tm = tmp_path / "map.json"
    tm.write_text(json.dumps({"ch06": "Polynomials"}), encoding="utf-8")
    got = rr._resolve_title_override("sample-algebra-2e-scan-ch06", None, str(tm))
    assert got == "Polynomials"


def test_title_map_no_match_returns_none(tmp_path):
    tm = tmp_path / "map.json"
    tm.write_text(json.dumps({"ch99": "X"}), encoding="utf-8")
    assert rr._resolve_title_override("scan-ch06", None, str(tm)) is None


def test_no_override_when_nothing_supplied():
    assert rr._resolve_title_override("scan-ch06", None, None) is None


# ---------------------------------------------------------------------------
# End-to-end --from-html re-render with --title.
# ---------------------------------------------------------------------------

_SAMPLE_HTML = """<!DOCTYPE html>
<html lang="en"><head><title>old</title></head><body>
<main id="main-content" role="main" class="dart-document">
<h1>188 Chapter 1 Foundations</h1>
<article role="doc-chapter" id="chap-1">
<h2>188 Chapter 1 Foundations</h2>
<section class="dart-section" data-dart-block-id="s1" data-dart-source="synthesized">
<h3 id="s1">1.1 Introduction to Whole Numbers</h3>
</section>
</article>
</main></body></html>
"""


def test_rerender_from_html_title_override(tmp_path):
    src = tmp_path / "sample-algebra-2e-scan-ch01_accessible.html"
    src.write_text(_SAMPLE_HTML, encoding="utf-8")
    out = tmp_path / "out.html"
    rc = rr.main([
        "--from-html", str(src),
        "--output", str(out),
        "--title", "Elementary Algebra 2e",
    ])
    assert rc == 0
    html = out.read_text(encoding="utf-8")
    assert "<h1>Elementary Algebra 2e</h1>" in html
    assert "<title>Elementary Algebra 2e</title>" in html
    # The running-header <h2> is neutralized out of the heading stream.
    assert "<h2>188 Chapter 1 Foundations</h2>" not in html


# ---------------------------------------------------------------------------
# SEMANTIK_OCR_CONFUSABLE_REPAIR — re-render preservation (design test 15).
#   --from-ir round-trips the additive repaired_text / ocr_repair keys onto the
#   rendered body + data-dart-repair stamp; --from-html preserves the already-
#   emitted data-dart-repair attributes across a reconstruction re-render.
# ---------------------------------------------------------------------------


def _repair_ir() -> dict:
    """A minimal cascade-IR doc: one chapter heading + one paragraph carrying an
    additive, replay-valid OCR-confusable repair (Vx -> √x)."""
    return {
        "pdf": "scan-ch01",
        "region_provenance": [
            {
                "region_index": 0,
                "region_kind": "heading",
                "role": "heading",
                "heading_text": "Chapter 1 Foundations",
                "level": 1,
                "first_raw_block_index": 0,
                "pages": [1],
                "raw_text": "Chapter 1 Foundations",
                "wcag_status": "passed",
            },
            {
                "region_index": 1,
                "region_kind": "paragraph",
                "role": "paragraph",
                "heading_text": None,
                "level": None,
                "first_raw_block_index": 1,
                "pages": [1],
                "raw_text": "Solve Vx = 4 for the value.",
                "wcag_status": "passed",
                "repaired_text": "Solve √x = 4 for the value.",
                "ocr_repair": {
                    "type": "ocr-confusable",
                    "n_edits": 1,
                    "edits": [
                        {"before": "V", "after": "√",
                         "rule_id": "v_to_radical", "token_index": 1}
                    ],
                },
            },
        ],
        "heading_tree": [],
        "exit_action": "ship_with_confidence",
        "wcag_status": "passed",
    }


def test_rerender_from_ir_roundtrips_repair_keys(tmp_path):
    ir_path = tmp_path / "scan-ch01_accessible.cascade_ir.json"
    ir_path.write_text(json.dumps(_repair_ir()), encoding="utf-8")
    out = tmp_path / "out.html"
    rc = rr.main(["--ir", str(ir_path), "--output", str(out)])
    assert rc == 0
    html = out.read_text(encoding="utf-8")
    # The repair annotation is re-stamped on the section opening tag.
    assert 'data-dart-repair="ocr-confusable"' in html
    assert 'data-dart-repair-count="1"' in html
    # The rendered body carries the REPAIRED glyph, not the OCR garble.
    assert "Solve √x = 4" in html
    assert "Solve Vx = 4" not in html


_REPAIR_HTML = """<!DOCTYPE html>
<html lang="en"><head><title>old</title></head><body>
<main id="main-content" role="main" class="dart-document">
<h1>Chapter 1 Foundations</h1>
<article role="doc-chapter" id="chap-1">
<h2>Chapter 1 Foundations</h2>
<section class="dart-section" data-dart-block-id="s1" data-dart-source="synthesized">
<h3 id="s1">Chapter 1 Foundations</h3>
</section>
<section class="dart-section" data-dart-block-id="s2" data-dart-source="synthesized" \
data-dart-block-role="paragraph" data-dart-repair="ocr-confusable" data-dart-repair-count="1">
<p id="s2" class="sr-only" hidden>paragraph</p>
<p>Solve √x = 4 for the value.</p>
</section>
</article>
</main></body></html>
"""

_REPAIR_SYNTH = {
    "sections": [
        {
            "section_id": "s2",
            "data": {
                "text": "Solve √x = 4 for the value.",
                "repair": {
                    "type": "ocr-confusable",
                    "n_edits": 1,
                    "original_text": "Solve Vx = 4 for the value.",
                },
            },
        }
    ]
}


def test_rerender_from_html_preserves_repair_attrs(tmp_path):
    src = tmp_path / "scan-ch01_accessible.html"
    src.write_text(_REPAIR_HTML, encoding="utf-8")
    synth = tmp_path / "scan-ch01_accessible_synthesized.json"
    synth.write_text(json.dumps(_REPAIR_SYNTH), encoding="utf-8")
    out = tmp_path / "out.html"
    rc = rr.main([
        "--from-html", str(src),
        "--synthesized", str(synth),
        "--output", str(out),
    ])
    assert rc == 0
    html = out.read_text(encoding="utf-8")
    # The data-dart-repair marker survives the --from-html reconstruction.
    assert 'data-dart-repair="ocr-confusable"' in html
    assert 'data-dart-repair-count="1"' in html
    # The repaired body bytes are carried verbatim (not reverted to the garble).
    assert "Solve √x = 4 for the value." in html
    assert "Solve Vx = 4" not in html

"""Standalone heading judge — figure-enrichment preservation (--source-html).

Regression (whole-book single-PDF reference corpus, 2026-07-22): the --apply
re-render rebuilds the accessible HTML from the layout sidecar via a fresh
``transform_document`` — but the sidecar carries the RAW GLM pages only, so
the VLM alt-text enrichment (``SEMANTIK_ALTTEXT_PROVIDER``) that lives
solely in the prior render is LOST and every VLM-captioned figure degrades
to the sr-only ``"Figure."`` placeholder.

Pins: ``--source-html`` restores degraded captions from the prior enriched
render; without it the (documented) degraded re-render still writes; a merge
failure never loses the render. The render seam is stubbed — no lib/semantik
adapter dependency, no seat, no GPU.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEMANTIK_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT), str(SEMANTIK_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from semantik_structure.glmocr import (  # noqa: E402
    heading_judge_standalone as hjs,
)
from semantik_structure.glmocr import lane  # noqa: E402

ENRICHED_PRIOR_HTML = (
    "<html><h3>Pending Level</h3>"
    '<section data-semantik-block-id="s2">'
    "<figure><figcaption>A network diagram showing interconnected nodes."
    "</figcaption></figure></section></html>"
)

DEGRADED_RERENDER_HTML = (
    "<html><h2>Pending Level</h2>"
    '<section data-semantik-block-id="s2">'
    '<figure><figcaption><span class="sr-only">Figure.</span>'
    "</figcaption></figure></section></html>"
)


def _layout(tmp_path: Path) -> Path:
    """A layout sidecar whose transform yields ZERO pending headings (one
    plain paragraph region) — the judge never POSTs."""
    layout = tmp_path / "ch01.glmocr_layout.json"
    layout.write_text(
        json.dumps({"pages": [{"page_no": 1, "regions": [{
            "index": 0,
            "native_label": "text",
            "bbox_2d": [0, 0, 100, 20],
            "content": "Plain paragraph prose.",
        }]}]}),
        encoding="utf-8",
    )
    return layout


def _stub_render(monkeypatch, html=DEGRADED_RERENDER_HTML):
    monkeypatch.setattr(
        lane, "render_accessible_html", lambda result, pdf_stem: html
    )


def test_source_html_restores_degraded_figure_caption(tmp_path, monkeypatch):
    _stub_render(monkeypatch)
    prior = tmp_path / "ch01_accessible.html"
    prior.write_text(ENRICHED_PRIOR_HTML, encoding="utf-8")

    out_dir = tmp_path / "out"
    report = hjs.run_standalone(
        _layout(tmp_path), out_dir=out_dir, apply=True, source_html=prior
    )

    written = (out_dir / "ch01_accessible.html").read_text(encoding="utf-8")
    # the VLM caption survives the re-render
    assert "A network diagram showing interconnected nodes." in written
    assert 'sr-only">Figure.' not in written
    # the judged heading level is untouched by the merge
    assert "<h2>Pending Level</h2>" in written
    assert report["figure_enrichment_restored"] == 1
    assert report["figure_enrichment_source"] == str(prior)
    # the prior render is an INPUT — never modified
    assert prior.read_text(encoding="utf-8") == ENRICHED_PRIOR_HTML


def test_without_source_html_rerender_writes_unmerged(tmp_path, monkeypatch):
    _stub_render(monkeypatch)
    out_dir = tmp_path / "out"
    report = hjs.run_standalone(
        _layout(tmp_path), out_dir=out_dir, apply=True
    )
    written = (out_dir / "ch01_accessible.html").read_text(encoding="utf-8")
    assert written == DEGRADED_RERENDER_HTML
    assert "figure_enrichment_restored" not in report


def test_missing_source_html_is_a_no_op(tmp_path, monkeypatch):
    _stub_render(monkeypatch)
    out_dir = tmp_path / "out"
    report = hjs.run_standalone(
        _layout(tmp_path), out_dir=out_dir, apply=True,
        source_html=tmp_path / "does-not-exist.html",
    )
    written = (out_dir / "ch01_accessible.html").read_text(encoding="utf-8")
    assert written == DEGRADED_RERENDER_HTML
    assert "figure_enrichment_restored" not in report


def test_merge_failure_never_loses_the_render(tmp_path, monkeypatch):
    _stub_render(monkeypatch)
    prior = tmp_path / "ch01_accessible.html"
    prior.write_text(ENRICHED_PRIOR_HTML, encoding="utf-8")

    import lib.semantik.figure_enrich_merge as fem

    def _boom(prior_html, judged_html):
        raise RuntimeError("merge exploded")

    monkeypatch.setattr(fem, "merge_figure_enrichment", _boom)

    out_dir = tmp_path / "out"
    report = hjs.run_standalone(
        _layout(tmp_path), out_dir=out_dir, apply=True, source_html=prior
    )
    written = (out_dir / "ch01_accessible.html").read_text(encoding="utf-8")
    assert written == DEGRADED_RERENDER_HTML  # unmerged, but never lost
    assert "figure_enrichment_restored" not in report


def test_cli_forwards_source_html(tmp_path, monkeypatch):
    seen = {}

    def _capture(layout, **kwargs):
        seen.update(kwargs)
        return {"stem": "ch01", "n_pending": 0}

    monkeypatch.setattr(hjs, "run_standalone", _capture)
    layout = _layout(tmp_path)
    prior = tmp_path / "ch01_accessible.html"
    prior.write_text("x", encoding="utf-8")
    rc = hjs.main([
        str(layout), "--apply", "--out", str(tmp_path / "out"),
        "--source-html", str(prior),
    ])
    assert rc == 0
    assert seen["source_html"] == prior
    assert seen["apply"] is True

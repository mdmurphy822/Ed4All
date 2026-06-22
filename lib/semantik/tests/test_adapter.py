"""P2a tests for the SemantiK v2 → Ed4All output-contract adapter.

Built + run with NO models / GPU. Exercises the adapter against a SYNTHETIC
cascade-result fixture and the REAL Ed4All contract validators
(``DartMarkersValidator``, ``source_refs`` helpers, ``SemanticStructureExtractor``).

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest lib/semantik/tests/test_adapter.py -q

The synthetic fixture (``_make_cascade_result``) is built FROM SCRATCH — no
vendored Semantic test fixture exists under ``SemantiK/`` (P0 vendored code
only). It mirrors the ``types.py`` region/feature-block shapes: each block
anchors to a RAW extracted-block document-order index (§3.3a), carries page
numbers, and the doc has ≥2 chapters + a figure + each of the three
``exit_action`` states.
"""

from __future__ import annotations

import re

import pytest

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _mint_sid,
    normalize_cascade_to_ed4all,
)


# ---------------------------------------------------------------------------
# Synthetic cascade-result fixture (built from scratch — §test item 4).
# ---------------------------------------------------------------------------


class _SyntheticCascadeResult:
    """Duck-typed stand-in for ``PipelineV2Result`` + the normalized IR.

    The real ``PipelineV2Result`` does NOT carry regions/feature_blocks in
    its serialized dict; the P3 seam attaches the normalized ``chapters`` IR
    derived from ``regions[i].feature_block_indices`` → ``feature_blocks[i].raw``.
    This synthetic result attaches the IR directly so P2a can validate the
    adapter without loading the cascade.
    """

    def __init__(self, *, exit_action: str):
        self.pdf = "sample_text_ch1.pdf"
        self.wcag_status = "passed"
        self.exit_action = exit_action
        self.theta_score = 0.91
        self.flags = [] if exit_action == "ship_with_confidence" else ["theta_low"]
        self.lane_used = "fast-lane"
        self.lang = "en"
        # ≥2 chapters, page numbers, a figure (§test item: fixture shape).
        self.chapters = [
            _AdapterChapter(
                title="Chapter 1: Foundations",
                blocks=[
                    _AdapterBlock(
                        html="<p>Algebra builds on arithmetic.</p>",
                        region_kind="paragraph",
                        raw_block_index=0,
                        raw_text="Algebra builds on arithmetic.",
                        heading_text="Introduction to Algebra",
                        pages=[1],
                        confidence=0.83,
                        block_role="body",
                        wcag_status="passed",
                    ),
                    _AdapterBlock(
                        html="<p>The order of operations is PEMDAS.</p>",
                        region_kind="paragraph",
                        raw_block_index=2,
                        raw_text="The order of operations is PEMDAS.",
                        heading_text="Order of Operations",
                        pages=[2, 3],
                        confidence=0.61,
                        block_role="body",
                        wcag_status="passed",
                    ),
                    _AdapterBlock(
                        html=(
                            '<figure><img src="fig1.png" alt="A number line">'
                            "<figcaption>Figure 1.1</figcaption></figure>"
                        ),
                        region_kind="figure",
                        raw_block_index=4,
                        raw_text="Figure 1.1 A number line from -5 to 5.",
                        heading_text="Number Line",
                        pages=[3],
                        confidence=1.0,  # bands to omitted
                        block_role="figure",
                        wcag_status="passed",
                        figure_alt="A number line from -5 to 5.",
                    ),
                    # A non-content (answer-key) heading that MUST be filtered
                    # so the chapter stays under the >40-section collapse.
                    _AdapterBlock(
                        html="<p>78 41. 900 42. 800</p>",
                        region_kind="paragraph",
                        raw_block_index=6,
                        raw_text="78 41. 900 42. 800 43. 700",
                        heading_text="78 41. 900 42. 800 43. 700",
                        pages=[4],
                        confidence=0.4,
                    ),
                ],
            ),
            _AdapterChapter(
                title="Chapter 2: Linear Equations",
                blocks=[
                    _AdapterBlock(
                        html="<p>Solve for x in 2x + 3 = 7.</p>",
                        region_kind="paragraph",
                        raw_block_index=10,
                        raw_text="Solve for x in 2x + 3 = 7.",
                        heading_text="Solving Linear Equations",
                        pages=[5],
                        confidence=0.79,
                        block_role="body",
                        wcag_status="passed",
                    ),
                    _AdapterBlock(
                        html="<p>A headingless prose block.</p>",
                        region_kind="paragraph",
                        raw_block_index=12,
                        raw_text="A headingless prose block continues here.",
                        heading_text=None,  # positional-fallback sid
                        pages=[6],
                        confidence=0.6,
                    ),
                ],
            ),
        ]


def _make_cascade_result(exit_action: str = "ship_with_confidence"):
    return _SyntheticCascadeResult(exit_action=exit_action)


@pytest.fixture
def result_html_and_sidecar():
    res = _make_cascade_result()
    out = normalize_cascade_to_ed4all(res, pdf_stem="sample_text_ch1")
    return out


# ---------------------------------------------------------------------------
# (a) §3.0 single-sid INVARIANT.
# ---------------------------------------------------------------------------


def test_a_single_sid_invariant(result_html_and_sidecar):
    """aria-labelledby == heading id == data-dart-block-id == sidecar
    section_id == #fragment == heading_slug(text), all from ONE mint fn."""
    from gui.services.source_page import heading_slug

    out = result_html_and_sidecar
    html = out["html"]

    # Harvest every section's (aria-labelledby, data-dart-block-id) and the
    # inner heading id.
    sections = re.findall(r"<section\b[^>]*>.*?</section>", html, re.DOTALL)
    assert sections, "no <section> wrappers emitted"

    sidecar_ids = {s["section_id"] for s in out["synthesized_sidecar"]["sections"]}
    assert sidecar_ids, "sidecar carried no sections"

    seen_block_ids = set()
    for sec in sections:
        aria = re.search(r'aria-labelledby="([^"]+)"', sec).group(1)
        block_id = re.search(r'data-dart-block-id="([^"]+)"', sec).group(1)
        heading_id = re.search(r'<h[1-6][^>]*\bid="([^"]+)"', sec).group(1)
        assert aria == block_id == heading_id, (
            f"sid divergence: aria={aria} block_id={block_id} hid={heading_id}"
        )
        # Parity with sidecar (§3.3 invariant).
        assert block_id in sidecar_ids, f"{block_id} not in sidecar id universe"
        seen_block_ids.add(block_id)

    # Every sidecar id appears in the HTML (full parity both directions).
    assert seen_block_ids == sidecar_ids

    # Heading-text-derived sids equal heading_slug(text) exactly.
    res = _make_cascade_result()
    intro_block = res.chapters[0].blocks[0]
    assert _mint_sid(intro_block) == heading_slug(intro_block.heading_text)
    # The #fragment for that sid is just #{sid}.
    assert f'href="#main-content"' in html  # skip-link fragment kept


def test_a_content_hash_mode_parity(monkeypatch):
    """§3.3 content-hash mode: HTML block_ids == sidecar section_ids, both
    16-hex hashes of the deterministic raw text."""
    monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "1")
    out = normalize_cascade_to_ed4all(
        _make_cascade_result(), pdf_stem="sample_text_ch1"
    )
    html = out["html"]
    html_ids = set(re.findall(r'data-dart-block-id="([^"]+)"', html))
    sidecar_ids = {
        s["section_id"] for s in out["synthesized_sidecar"]["sections"]
    }
    assert html_ids == sidecar_ids
    assert all(re.fullmatch(r"[0-9a-f]{16}", i) for i in html_ids)


# ---------------------------------------------------------------------------
# (b) DartMarkersValidator passes on adapter HTML.
# ---------------------------------------------------------------------------


def test_b_dart_markers_validator_passes(result_html_and_sidecar):
    from lib.validators.dart_markers import DartMarkersValidator

    out = result_html_and_sidecar
    res = DartMarkersValidator().validate({"html_content": out["html"]})
    critical = [i for i in res.issues if i.severity == "critical"]
    assert res.passed, f"dart_markers failed: {[i.code for i in critical]}"
    assert not critical
    # No empty data-dart-source anywhere.
    assert 'data-dart-source=""' not in out["html"]
    assert "data-dart-source=\"synthesized\"" in out["html"]


# ---------------------------------------------------------------------------
# (c) sourceId regex + source_refs parity.
# ---------------------------------------------------------------------------


def test_c_source_ids_match_regex_and_resolve(result_html_and_sidecar):
    from lib.validators.source_refs import SOURCE_ID_RE

    out = result_html_and_sidecar
    slug = out["slug"]
    sidecar_ids = {
        s["section_id"] for s in out["synthesized_sidecar"]["sections"]
    }
    # Every emitted block_id forms a valid sourceId AND resolves against the
    # sidecar id universe.
    html_ids = set(re.findall(r'data-dart-block-id="([^"]+)"', out["html"]))
    assert html_ids
    for bid in html_ids:
        source_id = f"dart:{slug}#{bid}"
        assert SOURCE_ID_RE.match(source_id), f"bad sourceId: {source_id}"
        assert bid in sidecar_ids, f"{bid} unresolved against sidecar"

    # Slug is the gentle transform (underscores preserved).
    assert slug == "sample_text_ch1"


# ---------------------------------------------------------------------------
# (d) SemanticStructureExtractor finds >=2 chapters, no >40-section collapse.
# ---------------------------------------------------------------------------


def test_d_structure_extractor_two_chapters_no_collapse(result_html_and_sidecar):
    from lib.semantic_structure_extractor.semantic_structure_extractor import (
        SemanticStructureExtractor,
        _STRUCTURE_COLLAPSE_SECTION_THRESHOLD,
    )

    out = result_html_and_sidecar
    extractor = SemanticStructureExtractor()
    structure = extractor.extract(out["html"])
    chapters = structure["chapters"]
    assert len(chapters) >= 2, f"expected >=2 chapters, got {len(chapters)}"
    for ch in chapters:
        secs = ch.get("sections", [])
        assert len(secs) <= _STRUCTURE_COLLAPSE_SECTION_THRESHOLD


# ---------------------------------------------------------------------------
# (e) exit_action → success / certification_status mapping.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exit_action,expected_success,expected_cert",
    [
        ("ship_with_confidence", True, "certified"),
        ("ship_with_flag", True, "flagged"),
        ("non_certified_stamp", True, "non_certified"),
        ("hard_error", False, "error"),
        (None, False, "error"),
    ],
)
def test_e_exit_action_mapping(exit_action, expected_success, expected_cert):
    res = _make_cascade_result(exit_action=exit_action or "ship_with_confidence")
    res.exit_action = exit_action  # overwrite (None case)
    out = normalize_cascade_to_ed4all(res, pdf_stem="sample_text_ch1")
    assert out["success"] is expected_success
    assert out["certification_status"] == expected_cert
    assert out["method"] == "semantik_v2"


def test_required_output_keys_present(result_html_and_sidecar):
    """The §3.7 / config/workflows.yaml:898 required keys are surfaced."""
    out = result_html_and_sidecar
    for key in ("html", "html_length", "success", "method", "word_count"):
        assert key in out
    assert out["html_length"] == len(out["html"])
    assert out["word_count"] > 0


def test_figure_and_page_provenance(result_html_and_sidecar):
    """Pages stamped as physical; figure_alt carried into the sidecar."""
    out = result_html_and_sidecar
    html = out["html"]
    assert 'data-dart-page-kind="physical"' in html
    assert 'data-dart-pages="2-3"' in html  # contiguous range collapse
    # 1.0-confidence figure omits data-dart-confidence (§3.6).
    fig_sec = re.search(
        r'<section[^>]*data-dart-block-role="figure"[^>]*>', html
    )
    assert fig_sec and "data-dart-confidence" not in fig_sec.group(0)
    # figure_alt rides along into the sidecar.
    alts = [
        s["data"].get("figure_alt")
        for s in out["synthesized_sidecar"]["sections"]
        if s["data"].get("figure_alt")
    ]
    assert "A number line from -5 to 5." in alts

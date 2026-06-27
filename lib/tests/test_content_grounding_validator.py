"""Wave 31 — ContentGroundingValidator tests.

Verifies the validator catches:

* Fully-grounded pages pass.
* Pages where ≥50% of paragraphs lack data-cf-source-ids fail critical.
* Unresolved source IDs (ID present but doesn't exist in staging) fail critical.
* Aggregate empty-page failure: ≥25% empty pages = critical, <25% = warning.

Hermetic: synthetic fixtures only. No corpus-specific identifiers.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.validators.content_grounding import ContentGroundingValidator  # noqa: E402

# ---------------------------------------------------------------------- #
# Helpers — build synthetic Courseforge pages + staging blocks
# ---------------------------------------------------------------------- #


def _make_para(words: int, source_ids: str = "") -> str:
    """Return a paragraph with exactly ``words`` words (non-trivial if ≥30)."""
    text = " ".join(f"word{i}" for i in range(words))
    attr = f' data-cf-source-ids="{source_ids}"' if source_ids else ""
    return f"<p{attr}>{text}</p>"


def _make_page(body: str) -> str:
    return (
        '<!DOCTYPE html><html lang="en"><head><title>P</title></head>'
        f'<body><main><h1>Week 1</h1>{body}</main></body></html>'
    )


def _build_staging(tmp_path: Path, block_ids: list) -> Path:
    """Build a staging dir with a synthesized sidecar declaring block IDs."""
    staging = tmp_path / "staging"
    staging.mkdir()
    # Write a minimal DART HTML with data-dart-block-id attributes.
    html = "<html><body>"
    for bid in block_ids:
        html += f'<section data-dart-block-id="{bid}"><p>Body content for {bid}</p></section>'
    html += "</body></html>"
    (staging / "sample.html").write_text(html, encoding="utf-8")
    return staging


# ---------------------------------------------------------------------- #
# Happy paths
# ---------------------------------------------------------------------- #


class TestHappyPath:
    def test_fully_grounded_page_passes(self, tmp_path):
        staging = _build_staging(tmp_path, ["s1", "s2", "s3"])
        # Page with 3 paragraphs, all carrying valid source IDs.
        body = "".join(_make_para(40, f"s{i}") for i in range(1, 4))
        page = tmp_path / "week_01_overview.html"
        page.write_text(_make_page(body), encoding="utf-8")

        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "staging_dir": str(staging),
        })
        assert result.passed is True
        assert result.score >= 0.9

    def test_valid_block_ids_pre_computed_override(self, tmp_path):
        body = "".join(_make_para(40, f"s{i}") for i in range(1, 4))
        page = tmp_path / "week_01.html"
        page.write_text(_make_page(body), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1", "s2", "s3"],
        })
        assert result.passed is True


# ---------------------------------------------------------------------- #
# Critical failure modes
# ---------------------------------------------------------------------- #


class TestUngroundedContent:
    def test_60_percent_ungrounded_fails_critical(self, tmp_path):
        # 5 paragraphs, only 2 carry data-cf-source-ids.
        body = (
            _make_para(40, "s1")
            + _make_para(40, "s2")
            + _make_para(40, "")
            + _make_para(40, "")
            + _make_para(40, "")
        )
        page = tmp_path / "week_01.html"
        page.write_text(_make_page(body), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1", "s2"],
        })
        assert result.passed is False
        codes = {i.code for i in result.issues}
        assert "PAGE_UNGROUNDED" in codes

    def test_three_of_five_ungrounded_is_critical(self, tmp_path):
        """3/5 = 60% > 50% threshold → critical."""
        body = (
            _make_para(40, "s1")
            + _make_para(40, "s2")
            + _make_para(40, "")
            + _make_para(40, "")
            + _make_para(40, "")
        )
        page = tmp_path / "week_01.html"
        page.write_text(_make_page(body), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1", "s2"],
        })
        assert result.passed is False


class TestUnresolvedSourceIds:
    def test_unknown_block_id_critical(self, tmp_path):
        staging = _build_staging(tmp_path, ["real_block"])
        body = _make_para(40, "ghost_block") + _make_para(40, "ghost_block_2") + _make_para(40, "ghost_block_3")
        page = tmp_path / "week_01.html"
        page.write_text(_make_page(body), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "staging_dir": str(staging),
        })
        assert result.passed is False
        codes = {i.code for i in result.issues}
        assert "UNRESOLVED_SOURCE_ID" in codes


# ---------------------------------------------------------------------- #
# Empty-page aggregate
# ---------------------------------------------------------------------- #


class TestEmptyPages:
    def test_all_empty_pages_critical(self, tmp_path):
        # Page with only short paragraphs (all < 30 words).
        body = "<p>Short.</p><p>Too short.</p>"
        pages = []
        for i in range(4):
            p = tmp_path / f"week_{i:02d}.html"
            p.write_text(_make_page(body), encoding="utf-8")
            pages.append(str(p))
        result = ContentGroundingValidator().validate({
            "page_paths": pages,
            "valid_block_ids": [],
        })
        assert result.passed is False
        codes = {i.code for i in result.issues}
        assert "AGGREGATE_EMPTY_PAGES" in codes

    def test_three_empty_of_48_is_warning(self, tmp_path):
        """3/48 empty pages (6%) is below the 25% threshold → warning only."""
        staging = _build_staging(tmp_path, ["s1"])
        pages = []
        full_body = "".join(_make_para(40, "s1") for _ in range(3))
        empty_body = "<p>Short.</p>"
        for i in range(45):
            p = tmp_path / f"week_full_{i:02d}.html"
            p.write_text(_make_page(full_body), encoding="utf-8")
            pages.append(str(p))
        for i in range(3):
            p = tmp_path / f"week_empty_{i:02d}.html"
            p.write_text(_make_page(empty_body), encoding="utf-8")
            pages.append(str(p))
        result = ContentGroundingValidator().validate({
            "page_paths": pages,
            "staging_dir": str(staging),
        })
        # 3/48 = 6.25% empty — below 25% threshold, so "SOME_EMPTY" warning,
        # not aggregate critical. Page-level critical may fire if 3 empty
        # pages → no, empty doesn't fire PAGE_UNGROUNDED.
        codes = {i.code for i in result.issues}
        assert "AGGREGATE_EMPTY_PAGES" not in codes
        assert "SOME_EMPTY_PAGES" in codes

    def test_twelve_empty_of_48_is_critical(self, tmp_path):
        """12/48 = 25% empty → critical."""
        staging = _build_staging(tmp_path, ["s1"])
        pages = []
        full_body = "".join(_make_para(40, "s1") for _ in range(3))
        empty_body = "<p>Short.</p>"
        for i in range(36):
            p = tmp_path / f"week_full_{i:02d}.html"
            p.write_text(_make_page(full_body), encoding="utf-8")
            pages.append(str(p))
        for i in range(12):
            p = tmp_path / f"week_empty_{i:02d}.html"
            p.write_text(_make_page(empty_body), encoding="utf-8")
            pages.append(str(p))
        result = ContentGroundingValidator().validate({
            "page_paths": pages,
            "staging_dir": str(staging),
        })
        assert result.passed is False
        codes = {i.code for i in result.issues}
        assert "AGGREGATE_EMPTY_PAGES" in codes


# ---------------------------------------------------------------------- #
# Ancestor-lookup for data-cf-source-ids
# ---------------------------------------------------------------------- #


class TestAncestorLookup:
    def test_source_id_on_ancestor_counts_as_grounded(self, tmp_path):
        """data-cf-source-ids on a wrapping <section> should cover its child <p>."""
        body = (
            '<section data-cf-source-ids="s1">'
            + _make_para(40, "")  # child paragraph without its own attr
            + '</section>'
        )
        page = tmp_path / "week_01.html"
        page.write_text(_make_page(body), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1"],
        })
        assert result.passed is True


# ---------------------------------------------------------------------- #
# Gate integration smoke
# ---------------------------------------------------------------------- #


class TestObjectivesPageExemption:
    """FIX 1 — the auto-generated objectives-list structural page is exempt.

    Its CO objective statements render as ungrounded ``<li>`` legitimately
    (objective statements carry no DART source block). The exemption is TIGHT:
    only the ``learning_objectives.html`` basename — week overview/summary pages
    stay checked (see ``test_week_overview_still_checked`` below).
    """

    def _objectives_body(self) -> str:
        # Five ungrounded objective <li> (no data-cf-source-ids) — would be a
        # 100%-ungrounded PAGE_UNGROUNDED critical on any non-exempt page.
        return "<ul>" + "".join(
            f"<li>{' '.join(f'objective{j}-word{i}' for i in range(40))}</li>"
            for j in range(5)
        ) + "</ul>"

    def test_learning_objectives_page_is_exempt(self, tmp_path):
        page = tmp_path / "learning_objectives.html"
        page.write_text(_make_page(self._objectives_body()), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1"],
        })
        # Exempt page → no PAGE_UNGROUNDED, passes for the right reason.
        assert result.passed is True
        codes = {i.code for i in result.issues}
        assert "PAGE_UNGROUNDED" not in codes
        # The exempt page is not counted as an empty content page either.
        assert "AGGREGATE_EMPTY_PAGES" not in codes

    def test_learning_objectives_under_course_overview_path_exempt(self, tmp_path):
        overview = tmp_path / "course_overview"
        overview.mkdir()
        page = overview / "learning_objectives.html"
        page.write_text(_make_page(self._objectives_body()), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1"],
        })
        assert result.passed is True
        codes = {i.code for i in result.issues}
        assert "PAGE_UNGROUNDED" not in codes

    def test_week_overview_still_checked_not_exempt(self, tmp_path):
        """CONTROL: a genuinely-ungrounded week_NN_overview page STILL fails.

        The exemption must NOT blanket *_overview / *_summary pages.
        """
        page = tmp_path / "week_03_overview.html"
        page.write_text(_make_page(self._objectives_body()), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1"],
        })
        assert result.passed is False
        codes = {i.code for i in result.issues}
        assert "PAGE_UNGROUNDED" in codes

    def test_summary_page_still_checked_not_exempt(self, tmp_path):
        """CONTROL: a *_summary page is NOT exempt and still fails ungrounded."""
        page = tmp_path / "week_03_summary.html"
        page.write_text(_make_page(self._objectives_body()), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1"],
        })
        assert result.passed is False
        assert "PAGE_UNGROUNDED" in {i.code for i in result.issues}


class TestChunkIdNamespace:
    """FIX 2 — chunk-id namespace source-ids resolve against the run chunkset.

    The key_terms feature stamps ``dart:{slug}#{course}_chunk_NNNNN`` ids that
    deep-link to real chunks. ``_resolve_valid_block_ids`` only harvests DART
    block ids, so these were categorically UNRESOLVED_SOURCE_ID. FIX 2 admits
    the chunk-id namespace by exact membership against the harvested chunkset.
    """

    def _write_chunks(self, tmp_path: Path, chunk_ids: list) -> Path:
        path = tmp_path / "dart_chunks" / "chunks.jsonl"
        path.parent.mkdir(parents=True)
        import json
        path.write_text(
            "\n".join(json.dumps({"id": c, "text": "body"}) for c in chunk_ids),
            encoding="utf-8",
        )
        return path

    def test_key_terms_chunk_id_resolves_against_chunkset(self, tmp_path):
        chunks = self._write_chunks(
            tmp_path, ["demo_chunk_00001", "demo_chunk_00002"]
        )
        # Vocab-card paragraph stamped with the canonical key_terms source-id.
        body = _make_para(40, "dart:demo#demo_chunk_00001")
        page = tmp_path / "week_01_key_terms.html"
        page.write_text(_make_page(body), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1"],  # DART block universe (irrelevant here)
            "dart_chunks_path": str(chunks),
        })
        # Resolves against the chunkset → passes for the right reason.
        assert result.passed is True
        codes = {i.code for i in result.issues}
        assert "UNRESOLVED_SOURCE_ID" not in codes
        assert "UNRESOLVED_CHUNK_SOURCE_ID" not in codes

    def test_valid_chunk_ids_precomputed_resolves(self, tmp_path):
        body = _make_para(40, "dart:demo#demo_chunk_00007")
        page = tmp_path / "week_01_key_terms.html"
        page.write_text(_make_page(body), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1"],
            "valid_chunk_ids": ["demo_chunk_00007"],
        })
        assert result.passed is True
        assert "UNRESOLVED_SOURCE_ID" not in {i.code for i in result.issues}

    def test_bogus_chunk_id_still_flags_critical_when_harvested(self, tmp_path):
        """CONTROL: a chunk-shaped id NOT in the harvested chunkset still fails.

        FIX 2 must not blanket-pass anything matching the chunk pattern — only
        ids that are REALLY in the run's chunkset resolve.
        """
        chunks = self._write_chunks(tmp_path, ["demo_chunk_00001"])
        body = (
            _make_para(40, "dart:demo#demo_chunk_99999")
            + _make_para(40, "dart:demo#demo_chunk_88888")
            + _make_para(40, "dart:demo#demo_chunk_77777")
        )
        page = tmp_path / "week_01_key_terms.html"
        page.write_text(_make_page(body), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1"],
            "dart_chunks_path": str(chunks),
        })
        assert result.passed is False
        assert "UNRESOLVED_SOURCE_ID" in {i.code for i in result.issues}

    def test_bogus_dart_block_id_still_flags_when_chunkset_present(self, tmp_path):
        """CONTROL: a genuinely-bogus DART block id STILL flags critical.

        Even with a chunkset harvested, a non-chunk-shaped unresolved id is a
        real grounding defect and must stay UNRESOLVED_SOURCE_ID critical.
        """
        chunks = self._write_chunks(tmp_path, ["demo_chunk_00001"])
        body = (
            _make_para(40, "ghost_block")
            + _make_para(40, "ghost_block_2")
            + _make_para(40, "ghost_block_3")
        )
        page = tmp_path / "week_01.html"
        page.write_text(_make_page(body), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["real_block"],
            "dart_chunks_path": str(chunks),
        })
        assert result.passed is False
        assert "UNRESOLVED_SOURCE_ID" in {i.code for i in result.issues}

    def test_chunk_id_demoted_to_warning_when_chunkset_unavailable(self, tmp_path):
        """Fallback: no harvestable chunkset → chunk-namespace id is a WARNING.

        The chunk-pattern unresolved demotes to UNRESOLVED_CHUNK_SOURCE_ID
        (warning), never blocking, while a real DART-block-id unresolved on the
        same page stays critical.
        """
        body = (
            _make_para(40, "dart:demo#demo_chunk_00001")  # chunk-shaped → warn
            + _make_para(40, "ghost_block")               # bogus block → crit
            + _make_para(40, "ghost_block_2")
            + _make_para(40, "ghost_block_3")
        )
        page = tmp_path / "week_01_key_terms.html"
        page.write_text(_make_page(body), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["real_block"],
            # No dart_chunks_path / valid_chunk_ids → harvest unavailable.
        })
        codes = {i.code for i in result.issues}
        # Chunk-namespace id demoted to a warning.
        assert "UNRESOLVED_CHUNK_SOURCE_ID" in codes
        # Bogus DART block ids still flag critical (detection unweakened).
        assert "UNRESOLVED_SOURCE_ID" in codes
        assert result.passed is False


class TestGateIntegration:
    def test_no_pages_returns_warning_not_crash(self):
        result = ContentGroundingValidator().validate({"page_paths": []})
        assert result.validator_name == "content_grounding"
        # Empty page list should pass gracefully with a skip warning.
        codes = {i.code for i in result.issues}
        assert "NO_PAGES_TO_SCAN" in codes

    def test_gate_result_has_score_and_gate_id(self, tmp_path):
        page = tmp_path / "w.html"
        page.write_text(_make_page(_make_para(40, "s1")), encoding="utf-8")
        result = ContentGroundingValidator().validate({
            "page_paths": [str(page)],
            "valid_block_ids": ["s1"],
            "gate_id": "content_grounding",
        })
        assert result.gate_id == "content_grounding"
        assert isinstance(result.score, float)

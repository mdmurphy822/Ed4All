"""Wave 9 — PageSourceRefValidator tests.

Covers the three failure modes the validator is designed to catch:

1. Emitted ``sourceId`` that does not resolve against the staging
   manifest → critical failure (the hallucination blocker).
2. Emitted ``sourceId`` that doesn't match the canonical pattern →
   critical failure.
3. Emitted refs when ``source_module_map.json`` is empty (and no
   valid_source_ids provided) → critical failure.

Plus the two happy paths:

- All emitted IDs resolve cleanly → gate passes.
- No emitted refs AND no populated map → gate passes (backward-compat).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.validators.source_refs import (  # noqa: E402
    PageSourceRefValidator,
    _iter_jsonld_source_ids,
    _iter_sidecar_block_ids,
)

# ---------------------------------------------------------------------- #
# Fixture helpers: synthesize staging + HTML inputs inline
# ---------------------------------------------------------------------- #


def _make_staging(tmp_path: Path, slug: str, block_ids: list, include_manifest: bool = True) -> Path:
    """Build a minimal staging_dir with one provenance_sidecar + manifest."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    sidecar_name = f"{slug}_synthesized.json"
    sidecar = staging_dir / sidecar_name
    sections = []
    for i, bid in enumerate(block_ids):
        sections.append({
            "section_id": f"s{i}",
            "section_type": "contacts",
            "section_title": f"Section {i}",
            "data": {
                "contacts": [
                    {"block_id": bid, "name": "Jane Doe"}
                ],
            },
        })
    sidecar.write_text(json.dumps({
        "campus_code": slug,
        "campus_name": slug.title(),
        "sections": sections,
    }))
    if include_manifest:
        manifest = {
            "run_id": "TEST_RUN",
            "course_name": "SAMPLE_101",
            "files": [
                {"path": f"{slug}.html", "role": "content"},
                {"path": sidecar_name, "role": "provenance_sidecar"},
            ],
        }
        (staging_dir / "staging_manifest.json").write_text(
            json.dumps(manifest)
        )
    return staging_dir


def _html_with_json_ld(source_ids: list) -> str:
    ld = {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": "SAMPLE_101",
        "weekNumber": 3,
        "moduleType": "content",
        "pageId": "week_03_content_01_x",
        "sourceReferences": [
            {"sourceId": sid, "role": "primary"} for sid in source_ids
        ],
    }
    attrs = ",".join(source_ids)
    return (
        '<!DOCTYPE html><html><head>'
        f'<script type="application/ld+json">{json.dumps(ld)}</script>'
        '</head>'
        f'<body><section data-cf-source-ids="{attrs}">'
        '<h2>Demo</h2></section></body></html>'
    )


def _html_with_attrs_only(source_ids: list, primary: str = "") -> str:
    joined = ",".join(source_ids)
    attr = f' data-cf-source-ids="{joined}"'
    if primary:
        attr += f' data-cf-source-primary="{primary}"'
    return (
        f'<!DOCTYPE html><html><body>'
        f'<section{attr}><h2>Demo</h2></section>'
        '</body></html>'
    )


# ---------------------------------------------------------------------- #
# Happy path
# ---------------------------------------------------------------------- #


class TestHappyPath:
    def test_valid_refs_with_staging_pass(self, tmp_path):
        staging = _make_staging(tmp_path, "science_of_learning", ["s0_c0", "s1_c0"])
        html = _html_with_json_ld(["dart:science_of_learning#s0_c0"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is True
        assert result.score == 1.0
        assert [i for i in result.issues if i.severity == "critical"] == []

    def test_empty_map_empty_refs_pass_backcompat(self, tmp_path):
        """Empty source_module_map.json + no emitted refs -> clean pass."""
        map_path = tmp_path / "source_module_map.json"
        map_path.write_text("{}")
        html = (
            '<!DOCTYPE html><html><body><section><h2>Demo</h2>'
            '</section></body></html>'
        )
        result = PageSourceRefValidator().validate({
            "source_module_map_path": str(map_path),
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is True
        assert result.score == 1.0

    def test_all_attrs_resolve_with_valid_source_ids_override(self):
        """Tests can seed valid_source_ids directly without a staging dir."""
        html = _html_with_attrs_only(
            ["dart:doc#s0", "dart:doc#s1"], primary="dart:doc#s0"
        )
        result = PageSourceRefValidator().validate({
            "valid_source_ids": ["dart:doc#s0", "dart:doc#s1"],
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is True

    def test_no_pages_at_all_passes_clean(self, tmp_path):
        """A run where nothing was generated yet -> gate is trivially clean."""
        staging = _make_staging(tmp_path, "x", ["s0_c0"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
        })
        assert result.passed is True
        assert result.score == 1.0


# ---------------------------------------------------------------------- #
# Bad sourceId: does not resolve against staging
# ---------------------------------------------------------------------- #


class TestUnresolvedSourceId:
    def test_unresolved_source_id_fails_critical(self, tmp_path):
        staging = _make_staging(tmp_path, "science_of_learning", ["s0_c0"])
        html = _html_with_json_ld(["dart:science_of_learning#not_a_block"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "html_contents": [{"path": "bad.html", "html": html}],
        })
        assert result.passed is False
        crit = [i for i in result.issues if i.severity == "critical"]
        codes = {i.code for i in crit}
        assert "UNRESOLVED_SOURCE_ID" in codes
        assert any("not_a_block" in i.message for i in crit)

    def test_wrong_document_slug_fails(self, tmp_path):
        staging = _make_staging(tmp_path, "science_of_learning", ["s0_c0"])
        html = _html_with_json_ld(["dart:other_doc#s0_c0"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "html_contents": [{"path": "bad.html", "html": html}],
        })
        assert result.passed is False

    def test_attr_only_emission_also_caught(self, tmp_path):
        """data-cf-source-ids without a JSON-LD block still gets validated."""
        staging = _make_staging(tmp_path, "x", ["s0_c0"])
        html = _html_with_attrs_only(["dart:x#ghost_id"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "html_contents": [{"path": "bad.html", "html": html}],
        })
        assert result.passed is False

    def test_mixed_valid_and_invalid_fails(self, tmp_path):
        staging = _make_staging(tmp_path, "x", ["s0_c0"])
        html = _html_with_json_ld(["dart:x#s0_c0", "dart:x#missing"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "html_contents": [{"path": "bad.html", "html": html}],
        })
        assert result.passed is False
        crit = [i for i in result.issues if i.code == "UNRESOLVED_SOURCE_ID"]
        assert len(crit) == 1
        # Score reflects 1/2 resolved.
        assert 0.0 < result.score < 1.0


# ---------------------------------------------------------------------- #
# Malformed shape
# ---------------------------------------------------------------------- #


class TestInvalidShape:
    def test_invalid_pattern_fails(self):
        html = _html_with_attrs_only(["foo-bar"])
        result = PageSourceRefValidator().validate({
            "valid_source_ids": ["foo-bar"],  # valid set contains it, but shape is bad
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is False
        assert any(
            i.code == "INVALID_SOURCE_ID_SHAPE" for i in result.issues
        )

    def test_uppercase_in_slug_fails(self):
        html = _html_with_attrs_only(["dart:SCIENCE#s0"])
        result = PageSourceRefValidator().validate({
            "valid_source_ids": ["dart:SCIENCE#s0"],
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is False

    def test_missing_separator_fails(self):
        html = _html_with_attrs_only(["dart:science_no_sep"])
        result = PageSourceRefValidator().validate({
            "valid_source_ids": ["dart:science_no_sep"],
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is False


# ---------------------------------------------------------------------- #
# Empty map but emitted refs -> critical
# ---------------------------------------------------------------------- #


class TestEmptyMapButEmittedRefs:
    def test_empty_map_with_emit_fails_critical(self, tmp_path):
        map_path = tmp_path / "source_module_map.json"
        map_path.write_text("{}")
        html = _html_with_json_ld(["dart:slug#s0_c0"])
        result = PageSourceRefValidator().validate({
            "source_module_map_path": str(map_path),
            "html_contents": [{"path": "oops.html", "html": html}],
        })
        assert result.passed is False
        codes = {i.code for i in result.issues}
        assert "UNEXPECTED_SOURCE_ID" in codes

    def test_missing_map_file_treated_as_empty(self, tmp_path):
        map_path = tmp_path / "does_not_exist.json"
        html = _html_with_json_ld(["dart:slug#s0_c0"])
        result = PageSourceRefValidator().validate({
            "source_module_map_path": str(map_path),
            "html_contents": [{"path": "oops.html", "html": html}],
        })
        assert result.passed is False


# ---------------------------------------------------------------------- #
# JSON-LD + sidecar walkers (unit)
# ---------------------------------------------------------------------- #


class TestJsonLdWalker:
    def test_walks_page_level_refs(self):
        data = {
            "sourceReferences": [
                {"sourceId": "dart:x#a", "role": "primary"},
                {"sourceId": "dart:x#b", "role": "contributing"},
            ]
        }
        assert sorted(_iter_jsonld_source_ids(data)) == ["dart:x#a", "dart:x#b"]

    def test_walks_section_level_refs(self):
        data = {
            "sections": [
                {
                    "sourceReferences": [
                        {"sourceId": "dart:x#c", "role": "primary"}
                    ]
                }
            ]
        }
        assert list(_iter_jsonld_source_ids(data)) == ["dart:x#c"]

    def test_walker_tolerates_missing_key(self):
        assert list(_iter_jsonld_source_ids({})) == []

    def test_walker_tolerates_malformed_entries(self):
        data = {"sourceReferences": [{}, None, "notadict", {"sourceId": ""}]}
        assert list(_iter_jsonld_source_ids(data)) == []


class TestSidecarWalker:
    def test_walks_campus_code_and_sections(self):
        sidecar = {
            "campus_code": "Science_of_Learning",
            "sections": [
                {"section_id": "s0", "data": {"contacts": [
                    {"block_id": "s0_c0"}
                ]}},
                {"section_id": "s1", "data": {"rows": [
                    {"block_id": "s1_r0"}
                ]}},
            ],
        }
        ids = sorted(_iter_sidecar_block_ids(sidecar))
        # Document slug is lower-cased via _slugify_doc.
        assert ids == [
            "dart:science_of_learning#s0",
            "dart:science_of_learning#s0_c0",
            "dart:science_of_learning#s1",
            "dart:science_of_learning#s1_r0",
        ]

    def test_walker_prefers_explicit_document_slug(self):
        sidecar = {
            "campus_code": "IGNORED",
            "document_slug": "override",
            "sections": [
                {"section_id": "s0", "data": {}},
            ],
        }
        ids = list(_iter_sidecar_block_ids(sidecar))
        assert "dart:override#s0" in ids

    def test_walker_returns_empty_when_no_slug(self):
        assert list(_iter_sidecar_block_ids({"sections": []})) == []

    def test_walker_handles_deep_nesting(self):
        sidecar = {
            "campus_code": "x",
            "sections": [{
                "section_id": "s0",
                "data": {
                    "pair_provenance": [
                        {"block_id": "s0_p0"},
                        {"nested": {"block_id": "s0_p1"}},
                    ]
                },
            }],
        }
        ids = sorted(_iter_sidecar_block_ids(sidecar))
        assert "dart:x#s0_p0" in ids
        assert "dart:x#s0_p1" in ids


# ---------------------------------------------------------------------- #
# File-reading path
# ---------------------------------------------------------------------- #


class TestFileReading:
    def test_reads_page_paths(self, tmp_path):
        staging = _make_staging(tmp_path, "x", ["s0_c0"])
        page_path = tmp_path / "page.html"
        page_path.write_text(_html_with_json_ld(["dart:x#s0_c0"]))
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "page_paths": [str(page_path)],
        })
        assert result.passed is True

    def test_missing_page_emits_warning(self, tmp_path):
        staging = _make_staging(tmp_path, "x", ["s0_c0"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "page_paths": [str(tmp_path / "does_not_exist.html")],
        })
        codes = {i.code for i in result.issues}
        assert "PAGE_NOT_FOUND" in codes
        # Warning doesn't block the gate.
        assert result.passed is True


# ---------------------------------------------------------------------- #
# Wave 27 CRITICAL-2: empty-emission warning on real runs
# ---------------------------------------------------------------------- #


class TestWave27EmptyEmitWarning:
    """Wave 27 turn-down: real textbook-to-course runs should always emit
    source-ids. Empty emit on a run that actually fed pages in surfaces
    as a WARNING (not a failure) so the regression shows up in gate
    output but legacy callers still pass.
    """

    def test_empty_emit_with_pages_emits_warning(self, tmp_path):
        html = (
            '<!DOCTYPE html><html><body><section><h2>Demo</h2>'
            '</section></body></html>'
        )
        result = PageSourceRefValidator().validate({
            "html_contents": [{"path": "page.html", "html": html}],
        })
        # Still passes — non-breaking change for legacy callers.
        assert result.passed is True
        codes = {i.code for i in result.issues}
        # ...but the Wave 27 warning is recorded.
        assert "EMPTY_SOURCE_REFS" in codes
        warnings = [
            i for i in result.issues
            if i.severity == "warning" and i.code == "EMPTY_SOURCE_REFS"
        ]
        assert warnings
        assert "data-cf-source-ids" in warnings[0].message

    def test_no_pages_no_warning(self, tmp_path):
        """Genuinely-legacy callers (no pages passed at all) stay silent."""
        staging = _make_staging(tmp_path, "x", ["s0_c0"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
        })
        codes = {i.code for i in result.issues}
        # No pages => no warning (classic backward-compat path).
        assert "EMPTY_SOURCE_REFS" not in codes
        assert result.passed is True

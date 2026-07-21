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
        html = _html_with_json_ld(["semantik:science_of_learning#s0_c0"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is True
        assert result.score == 1.0
        assert [i for i in result.issues if i.severity == "critical"] == []

    def test_empty_map_empty_refs_pass_backcompat(self, tmp_path):
        """Empty source_module_map.json + Wave-27 boilerplate carve-out
        (empty-string ``data-cf-source-ids=""``) -> clean pass.

        Wave4-I10: a page with NO ``data-cf-source-ids`` attr at all
        now trips EMPTY_SOURCE_REFS critical; the Wave-27 contract
        carve-out is to stamp the empty-string attribute explicitly
        to mark "boilerplate, no source provenance".
        """
        map_path = tmp_path / "source_module_map.json"
        map_path.write_text("{}")
        html = (
            '<!DOCTYPE html><html><body>'
            '<section data-cf-source-ids=""><h2>Demo</h2>'
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
            ["semantik:doc#s0", "semantik:doc#s1"], primary="semantik:doc#s0"
        )
        result = PageSourceRefValidator().validate({
            "valid_source_ids": ["semantik:doc#s0", "semantik:doc#s1"],
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
        html = _html_with_json_ld(["semantik:science_of_learning#not_a_block"])
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
        html = _html_with_json_ld(["semantik:other_doc#s0_c0"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "html_contents": [{"path": "bad.html", "html": html}],
        })
        assert result.passed is False

    def test_attr_only_emission_also_caught(self, tmp_path):
        """data-cf-source-ids without a JSON-LD block still gets validated."""
        staging = _make_staging(tmp_path, "x", ["s0_c0"])
        html = _html_with_attrs_only(["semantik:x#ghost_id"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "html_contents": [{"path": "bad.html", "html": html}],
        })
        assert result.passed is False

    def test_mixed_valid_and_invalid_fails(self, tmp_path):
        staging = _make_staging(tmp_path, "x", ["s0_c0"])
        html = _html_with_json_ld(["semantik:x#s0_c0", "semantik:x#missing"])
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
        html = _html_with_attrs_only(["semantik:SCIENCE#s0"])
        result = PageSourceRefValidator().validate({
            "valid_source_ids": ["semantik:SCIENCE#s0"],
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is False

    def test_missing_separator_fails(self):
        html = _html_with_attrs_only(["semantik:science_no_sep"])
        result = PageSourceRefValidator().validate({
            "valid_source_ids": ["semantik:science_no_sep"],
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
        html = _html_with_json_ld(["semantik:slug#s0_c0"])
        result = PageSourceRefValidator().validate({
            "source_module_map_path": str(map_path),
            "html_contents": [{"path": "oops.html", "html": html}],
        })
        assert result.passed is False
        codes = {i.code for i in result.issues}
        assert "UNEXPECTED_SOURCE_ID" in codes

    def test_missing_map_file_treated_as_empty(self, tmp_path):
        map_path = tmp_path / "does_not_exist.json"
        html = _html_with_json_ld(["semantik:slug#s0_c0"])
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
                {"sourceId": "semantik:x#a", "role": "primary"},
                {"sourceId": "semantik:x#b", "role": "contributing"},
            ]
        }
        assert sorted(_iter_jsonld_source_ids(data)) == ["semantik:x#a", "semantik:x#b"]

    def test_walks_section_level_refs(self):
        data = {
            "sections": [
                {
                    "sourceReferences": [
                        {"sourceId": "semantik:x#c", "role": "primary"}
                    ]
                }
            ]
        }
        assert list(_iter_jsonld_source_ids(data)) == ["semantik:x#c"]

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
        # Document slug is lower-cased via _slugify_doc. Stage-3 dual valid-
        # universe: every id is yielded under BOTH the legacy ``dart:`` and the
        # ratified ``semantik:`` prefix so a freshly-emitted semantik: sourceId
        # and a legacy dart: sourceId both resolve against the same sidecar.
        assert ids == [
            "dart:science_of_learning#s0",
            "dart:science_of_learning#s0_c0",
            "dart:science_of_learning#s1",
            "dart:science_of_learning#s1_r0",
            "semantik:science_of_learning#s0",
            "semantik:science_of_learning#s0_c0",
            "semantik:science_of_learning#s1",
            "semantik:science_of_learning#s1_r0",
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
        assert "semantik:override#s0" in ids

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
        assert "semantik:x#s0_p0" in ids
        assert "semantik:x#s0_p1" in ids


# ---------------------------------------------------------------------- #
# File-reading path
# ---------------------------------------------------------------------- #


class TestFileReading:
    def test_reads_page_paths(self, tmp_path):
        staging = _make_staging(tmp_path, "x", ["s0_c0"])
        page_path = tmp_path / "page.html"
        page_path.write_text(_html_with_json_ld(["semantik:x#s0_c0"]))
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


class TestSourceRefsManifestMissing:
    """C5 audit fix: when ``staging_dir`` is provided but the manifest
    + sidecars produce zero valid IDs, AND the emitter actually stamped
    sourceIds into HTML, fail closed rather than silently accept any ID
    (the legacy ``valid_ids and sid not in valid_ids`` short-circuit).
    """

    def test_empty_staging_dir_with_emitted_ids_fails_closed(self, tmp_path):
        # staging_dir exists but is completely empty — no manifest, no
        # sidecars. Harvested valid_ids will be empty.
        empty_staging = tmp_path / "staging_empty"
        empty_staging.mkdir()
        html = _html_with_json_ld(["semantik:slug#s0_c0"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(empty_staging),
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is False
        codes = {i.code for i in result.issues}
        assert "SOURCE_REFS_MANIFEST_MISSING" in codes
        crit = [
            i for i in result.issues
            if i.code == "SOURCE_REFS_MANIFEST_MISSING"
        ]
        assert crit and crit[0].severity == "critical"
        msg = crit[0].message
        assert str(empty_staging) in msg
        assert "stage_semantik_outputs" in msg
        assert "staging_manifest.json" in msg

    def test_missing_manifest_no_sidecars_fails_closed(self, tmp_path):
        """Staging dir with no manifest AND no sidecars + emitted IDs."""
        staging = tmp_path / "staging_no_manifest"
        staging.mkdir()
        # Create unrelated junk so dir exists but no sidecars resolve.
        (staging / "readme.txt").write_text("nothing useful here")
        html = _html_with_attrs_only(["semantik:slug#s0_c0"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is False
        codes = {i.code for i in result.issues}
        assert "SOURCE_REFS_MANIFEST_MISSING" in codes

    def test_explicit_valid_source_ids_empty_list_does_not_trigger(self, tmp_path):
        """Callers that explicitly seed ``valid_source_ids=[]`` are
        intentionally telling the gate "no refs expected". The new
        fail-closed branch must not mistake that for the silent-degrade
        case (which is detected via missing ``staging_dir``).
        """
        html = _html_with_attrs_only(["semantik:doc#s0"])
        result = PageSourceRefValidator().validate({
            "valid_source_ids": [],
            "html_contents": [{"path": "page.html", "html": html}],
        })
        # No SOURCE_REFS_MANIFEST_MISSING; the existing
        # UNRESOLVED_SOURCE_ID logic already fires (correct legacy
        # behavior — caller said "no IDs are valid").
        codes = {i.code for i in result.issues}
        assert "SOURCE_REFS_MANIFEST_MISSING" not in codes

    def test_no_emitted_ids_does_not_trigger(self, tmp_path):
        """Empty staging + no emitted IDs should NOT trip
        SOURCE_REFS_MANIFEST_MISSING (the fail-closed branch this
        class covers).

        Wave4-I10: a page with NO ``data-cf-source-ids`` attr now
        trips EMPTY_SOURCE_REFS critical — this test's contract is
        purely about the manifest-missing branch, so use the Wave-27
        empty-string carve-out to keep the EMPTY_SOURCE_REFS path
        clean and isolate the manifest-missing check.
        """
        empty_staging = tmp_path / "staging_empty"
        empty_staging.mkdir()
        result = PageSourceRefValidator().validate({
            "staging_dir": str(empty_staging),
            "html_contents": [{
                "path": "page.html",
                "html": '<html><body><div data-cf-source-ids=""></div></body></html>',
            }],
        })
        codes = {i.code for i in result.issues}
        assert "SOURCE_REFS_MANIFEST_MISSING" not in codes
        assert "EMPTY_SOURCE_REFS" not in codes
        assert result.passed is True


class TestWave4I10EmptySourceRefsCritical:
    """Wave4-I10 (follows Wave4-W27 `ffe517d`): EMPTY_SOURCE_REFS is now
    critical-severity. A page with ZERO ``data-cf-source-ids`` attrs
    fails the gate. Empty-string ``data-cf-source-ids=""`` is the
    explicit Wave-27 boilerplate carve-out and must still pass.
    """

    def test_pages_with_no_source_ids_attr_fail_critical(self, tmp_path):
        """No data-cf-source-ids on any wrapper -> critical, gate fails."""
        html = (
            '<!DOCTYPE html><html><body><section><h2>Demo</h2>'
            '</section></body></html>'
        )
        result = PageSourceRefValidator().validate({
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is False
        crit = [
            i for i in result.issues
            if i.severity == "critical" and i.code == "EMPTY_SOURCE_REFS"
        ]
        assert crit, "expected critical EMPTY_SOURCE_REFS GateIssue"
        assert "data-cf-source-ids" in crit[0].message
        # Wave-27 carve-out is documented in the message.
        assert "empty-string" in crit[0].message.lower()

    def test_empty_string_attr_passes_wave27_carveout(self, tmp_path):
        """`data-cf-source-ids=""` (empty string) -> passes, no GateIssue.

        Wave-27 boilerplate contract: content blocks without source
        provenance still stamp the attribute, just with an empty value.
        """
        html = (
            '<!DOCTYPE html><html><body>'
            '<section data-cf-source-ids=""><h2>Boilerplate</h2>'
            '</section></body></html>'
        )
        result = PageSourceRefValidator().validate({
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is True
        codes = {i.code for i in result.issues}
        assert "EMPTY_SOURCE_REFS" not in codes

    def test_populated_attr_passes(self, tmp_path):
        """Populated `data-cf-source-ids="semantik:doc#s0"` -> passes clean."""
        html = _html_with_attrs_only(["semantik:doc#s0"])
        result = PageSourceRefValidator().validate({
            "valid_source_ids": ["semantik:doc#s0"],
            "html_contents": [{"path": "page.html", "html": html}],
        })
        assert result.passed is True
        codes = {i.code for i in result.issues}
        assert "EMPTY_SOURCE_REFS" not in codes

    def test_no_pages_no_critical(self, tmp_path):
        """Genuinely-legacy callers (no pages passed at all) stay silent.

        Backward-compat: workflows that never pass pages (e.g. dry-run,
        staging-only checks) must not trip EMPTY_SOURCE_REFS.
        """
        staging = _make_staging(tmp_path, "x", ["s0_c0"])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
        })
        codes = {i.code for i in result.issues}
        assert "EMPTY_SOURCE_REFS" not in codes
        assert result.passed is True


class TestWave4I10WorkflowConfig:
    """Wave4-I10: ensure the workflows.yaml gate row keeps severity=critical
    + on_fail=block so the validator-level critical actually blocks the
    workflow.
    """

    def test_source_refs_gate_is_critical_block(self):
        import yaml

        cfg_path = PROJECT_ROOT / "config" / "workflows.yaml"
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        # Find the content_generation phase under textbook_to_course.
        tbc = cfg["workflows"]["textbook_to_course"]
        phases = tbc["phases"]
        content_gen = next(
            p for p in phases if p["name"] == "content_generation"
        )
        gate = next(
            g for g in content_gen["validation_gates"]
            if g["gate_id"] == "source_refs"
        )
        assert gate["validator"].endswith("PageSourceRefValidator")
        assert gate["severity"] == "critical"
        assert gate["behavior"]["on_fail"] == "block"
        assert gate["threshold"]["max_critical_issues"] == 0


# ---------------------------------------------------------------------- #
# SemantiK converter sidecar shape regression.
#
# The SemantiK converter writes a top-level ``slug`` field derived from
# the document *title* (e.g. ``01-accessibility-foundations``) and NO
# ``campus_code`` / ``document_slug`` field. The canonical sourceId slug,
# however, is derived from the sidecar FILENAME stem
# (``01_accessibility_foundations_accessible``) -- the rule shared by the
# source-router, content-grounding validator, and content-generator
# emitter. Reading the internal field mints IDs that never match the
# emitter, so every ``--skip-conversion`` / pre-converted run trips
# SOURCE_REFS_MANIFEST_MISSING. These tests build a staging dir in the
# converter's sidecar shape and assert the gate resolves cleanly.
# ---------------------------------------------------------------------- #


def _make_converter_shaped_staging(
    tmp_path: Path,
    file_stem: str,
    internal_slug: str,
    section_ids: list,
    include_manifest: bool = True,
) -> Path:
    """Build a staging dir whose sidecar matches the SemantiK converter shape.

    Mirrors the synthesized-sidecar builder: top-level ``slug``
    (title-derived, hyphenated), ``title``,
    ``source_pdf``, ``sections[]`` keyed on ``section_id`` -- and NO
    ``campus_code`` / ``document_slug`` field.
    """
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    sidecar_name = f"{file_stem}_synthesized.json"
    sections = []
    for sid in section_ids:
        sections.append({
            "section_id": sid,
            "section_title": f"Section {sid}",
            "section_type": "paragraph-group",
            "page_range": [1, 3],
            "provenance": {
                "sources": ["pdftotext"],
                "strategy": "heuristic",
                "confidence": 0.5,
            },
            "data": {"text": "body", "head_block_id": f"{sid}_b0"},
        })
    (staging_dir / sidecar_name).write_text(json.dumps({
        "slug": internal_slug,          # title-derived, deliberately != stem
        "title": internal_slug.replace("-", " ").title(),
        "source_pdf": f"/corpus/{file_stem}.pdf",
        "sections": sections,
        "document_provenance": {"extractors_used": ["pdftotext"]},
    }))
    if include_manifest:
        (staging_dir / "staging_manifest.json").write_text(json.dumps({
            "run_id": "TEST_RUN",
            "course_name": "SAMPLE_101",
            "files": [
                {"path": f"{file_stem}.html", "role": "content"},
                {"path": sidecar_name, "role": "provenance_sidecar"},
            ],
        }))
    return staging_dir


class TestConverterShapedSidecarHarvest:
    """A1 regression: filename-derived slug, NOT the internal field."""

    def test_canonical_slug_comes_from_filename_not_internal_field(
        self, tmp_path
    ):
        # file stem -> canonical slug
        # ``01_accessibility_foundations_accessible``; internal slug is the
        # title-derived ``01-accessibility-foundations`` which must be
        # IGNORED for sourceId minting.
        staging = _make_converter_shaped_staging(
            tmp_path,
            file_stem="01_accessibility_foundations_accessible",
            internal_slug="01-accessibility-foundations",
            section_ids=["s1", "s2", "s3"],
        )
        valid = PageSourceRefValidator()._collect_valid_ids(
            {"staging_dir": str(staging)}
        )
        assert (
            "semantik:01_accessibility_foundations_accessible#s1" in valid
        )
        # The title-derived internal slug must NOT appear.
        assert not any(
            sid.startswith("semantik:01-accessibility-foundations#")
            for sid in valid
        )

    def test_gate_passes_for_converter_sidecar_skip_conversion_path(
        self, tmp_path
    ):
        """End-to-end: emitter stamps the filename-derived sourceId; the
        gate must resolve it and pass (the exact case that failed
        real --skip-conversion runs)."""
        staging = _make_converter_shaped_staging(
            tmp_path,
            file_stem="02_aria_and_component_patterns_accessible",
            internal_slug="02-aria-and-component-patterns",
            section_ids=["s1", "s2"],
        )
        emitted = "semantik:02_aria_and_component_patterns_accessible#s1"
        html = _html_with_json_ld([emitted])
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "html_contents": [{"path": "page.html", "html": html}],
        })
        codes = {i.code for i in result.issues}
        assert "SOURCE_REFS_MANIFEST_MISSING" not in codes
        assert "UNRESOLVED_SOURCE_ID" not in codes
        assert result.passed is True

    def test_gate_resolves_via_glob_fallback_without_manifest(
        self, tmp_path
    ):
        """No staging_manifest.json -> harvester discovers sidecars by
        glob; filename-derived slug still applies."""
        staging = _make_converter_shaped_staging(
            tmp_path,
            file_stem="03_visual_design_principles_accessible",
            internal_slug="03-visual-design-principles",
            section_ids=["s1"],
            include_manifest=False,
        )
        emitted = "semantik:03_visual_design_principles_accessible#s1"
        result = PageSourceRefValidator().validate({
            "staging_dir": str(staging),
            "html_contents": [
                {"path": "p.html", "html": _html_with_json_ld([emitted])}
            ],
        })
        codes = {i.code for i in result.issues}
        assert "SOURCE_REFS_MANIFEST_MISSING" not in codes
        assert result.passed is True

    def test_empty_dir_still_fails_closed_actionably(self, tmp_path):
        """The fail-closed contract is preserved: a staging dir with no
        sidecars + emitted IDs still trips an actionable
        SOURCE_REFS_MANIFEST_MISSING."""
        empty = tmp_path / "staging_empty"
        empty.mkdir()
        emitted = "semantik:some_doc_accessible#s1"
        result = PageSourceRefValidator().validate({
            "staging_dir": str(empty),
            "html_contents": [
                {"path": "p.html", "html": _html_with_json_ld([emitted])}
            ],
        })
        assert result.passed is False
        crit = [
            i for i in result.issues
            if i.code == "SOURCE_REFS_MANIFEST_MISSING"
        ]
        assert crit and crit[0].severity == "critical"
        # Actionable: names the dir, the manifest, and the upstream phase.
        msg = crit[0].message
        assert str(empty) in msg
        assert "staging_manifest.json" in msg
        assert "stage_semantik_outputs" in msg

    def test_legacy_internal_slug_still_resolves_without_override(self):
        """Back-compat: a direct ``_iter_sidecar_block_ids`` call WITHOUT
        a slug override still falls back to the legacy internal
        campus_code/document_slug resolution (multi_source_interpreter
        sidecars + existing unit-test callers)."""
        legacy = {
            "campus_code": "Science_of_Learning",
            "sections": [
                {"section_id": "s1", "data": {
                    "contacts": [{"block_id": "s1_c0"}]
                }},
            ],
        }
        ids = set(_iter_sidecar_block_ids(legacy))
        assert "semantik:science_of_learning#s1" in ids
        assert "semantik:science_of_learning#s1_c0" in ids

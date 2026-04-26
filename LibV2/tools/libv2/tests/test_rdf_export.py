"""Tests for the libv2 export-rdf CLI (Phase 1.5).

Exercises ``LibV2/tools/libv2/rdf_export.py::export_course`` end-to-end:
reads real per-course JSON artifacts under ``LibV2/courses/<slug>/``,
applies the matching ``schemas/context/*_v1.jsonld`` @context, and
asserts that Turtle (and other) serializations land on disk with
non-zero triple counts.

The export is read-only against archived courses — we use the
rdf-shacl-551-2 corpus as the realistic fixture and a temp output
directory so we don't pollute the repo state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pyld = pytest.importorskip("pyld")
rdflib = pytest.importorskip("rdflib")

from LibV2.tools.libv2.rdf_export import (
    ExportResult,
    export_course,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
LIBV2_ROOT = PROJECT_ROOT / "LibV2"
FIXTURE_SLUG = "rdf-shacl-551-2"
FIXTURE_DIR = LIBV2_ROOT / "courses" / FIXTURE_SLUG


def _require_fixture():
    if not FIXTURE_DIR.is_dir():
        pytest.skip(
            f"rdf-shacl-551-2 fixture missing at {FIXTURE_DIR}; "
            "Phase 1.5 tests need the canonical corpus."
        )


# ---------------------------------------------------------------------------
# Happy path: turtle export against the real fixture
# ---------------------------------------------------------------------------


class TestExportCourseTurtle:
    def test_emits_one_file_per_present_artifact(self, tmp_path: Path):
        _require_fixture()
        results = export_course(LIBV2_ROOT, FIXTURE_SLUG, tmp_path)
        # rdf-shacl-551-2 has all three known artifacts present.
        assert len(results) == 3
        assert all(isinstance(r, ExportResult) for r in results)
        for r in results:
            assert Path(r.output_path).is_file()
            assert Path(r.output_path).suffix == ".ttl"

    def test_emitted_files_parse_back_via_rdflib(self, tmp_path: Path):
        _require_fixture()
        results = export_course(LIBV2_ROOT, FIXTURE_SLUG, tmp_path)
        for r in results:
            g = rdflib.Graph()
            g.parse(r.output_path, format="turtle")
            assert len(g) == r.triple_count, (
                f"{r.artifact_relpath}: re-parsed triple count drifted "
                f"({len(g)}) from declared count ({r.triple_count})."
            )

    def test_concept_graph_export_has_substantial_triple_count(self, tmp_path: Path):
        _require_fixture()
        results = export_course(LIBV2_ROOT, FIXTURE_SLUG, tmp_path)
        # concept_graph_semantic empirically yields ~101k triples on rdf-shacl-551-2.
        cg = next(r for r in results if "concept_graph_semantic" in r.artifact_relpath)
        assert cg.triple_count > 1000, (
            f"concept_graph_semantic should yield >>1000 triples; got {cg.triple_count}"
        )

    def test_pedagogy_graph_export_has_substantial_triple_count(self, tmp_path: Path):
        _require_fixture()
        results = export_course(LIBV2_ROOT, FIXTURE_SLUG, tmp_path)
        # pedagogy_graph empirically yields ~47k triples.
        pg = next(r for r in results if "pedagogy_graph" in r.artifact_relpath)
        assert pg.triple_count > 1000

    def test_course_export_has_learning_outcome_triples(self, tmp_path: Path):
        _require_fixture()
        results = export_course(LIBV2_ROOT, FIXTURE_SLUG, tmp_path)
        course = next(r for r in results if r.artifact_relpath == "course.json")
        g = rdflib.Graph()
        g.parse(course.output_path, format="turtle")
        # course.json's LO array is keyed via ed4all:hasLearningObjective.
        # The export does NOT inject per-LO ``@type`` (round-trip tests
        # do; the export honors what's on disk), so the count we can
        # rely on is the predicate-level fan-out from the course node.
        from rdflib import URIRef
        has_lo = URIRef("https://ed4all.io/vocab/hasLearningObjective")
        lo_triples = list(g.triples((None, has_lo, None)))
        assert len(lo_triples) > 5, (
            f"course.ttl should carry multiple ed4all:hasLearningObjective "
            f"triples; got {len(lo_triples)}"
        )


# ---------------------------------------------------------------------------
# Format coverage
# ---------------------------------------------------------------------------


class TestExportFormats:
    @pytest.mark.parametrize("fmt,expected_ext", [
        ("turtle", ".ttl"),
        ("trig", ".trig"),
        ("nquads", ".nq"),
        ("ntriples", ".nt"),
    ])
    def test_format_yields_expected_extension(self, tmp_path: Path, fmt: str, expected_ext: str):
        _require_fixture()
        results = export_course(LIBV2_ROOT, FIXTURE_SLUG, tmp_path, output_format=fmt)
        assert all(Path(r.output_path).suffix == expected_ext for r in results)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestExportErrors:
    def test_missing_course_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            export_course(LIBV2_ROOT, "no-such-course-slug", tmp_path)

    def test_output_directory_created_if_missing(self, tmp_path: Path):
        _require_fixture()
        out_dir = tmp_path / "deeply" / "nested" / "out"
        assert not out_dir.exists()
        results = export_course(LIBV2_ROOT, FIXTURE_SLUG, out_dir)
        assert out_dir.is_dir()
        assert len(results) > 0

    def test_missing_project_tree_raises(self, tmp_path: Path):
        # Synthetic course outside the project tree: no schemas/context/
        # is reachable, so export must fail loudly rather than silently
        # producing zero output.
        synthetic_libv2 = tmp_path / "lv2"
        course_dir = synthetic_libv2 / "courses" / "synthetic"
        course_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="schemas/context/"):
            export_course(synthetic_libv2, "synthetic", tmp_path / "out")


# ---------------------------------------------------------------------------
# ExportResult shape
# ---------------------------------------------------------------------------


class TestExportResultShape:
    def test_to_dict_carries_required_fields(self, tmp_path: Path):
        _require_fixture()
        results = export_course(LIBV2_ROOT, FIXTURE_SLUG, tmp_path)
        for r in results:
            d = r.to_dict()
            assert set(d.keys()) == {"artifact", "context", "output", "triples"}
            assert d["triples"] == r.triple_count

"""
Tests for per-week learningObjectives specificity in generate_course.py
(Worker H — upstream fix for the Trainforge LO-fanout defect).

What this defends against:
    Before the fix, every generated page embedded a JSON-LD
    ``learningObjectives`` block derived from the week-local ``objectives``
    list in the course_data JSON, and those week-local IDs
    (``W01-CO-01`` etc.) all collapsed to the same four IDs after
    Trainforge's week-prefix normalization. Result:
    ``outcome_reverse_coverage == 0.143`` and 24 of 28 declared outcomes
    uncovered.

These tests exercise the deterministic LO-selection logic and the
generated JSON-LD for representative weeks. A failure here reproduces
the defect.
"""

import json
import sys
from pathlib import Path

import pytest

# The scripts directory sits one level up from this tests/ dir.
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_course import (  # noqa: E402
    generate_week,
    load_canonical_objectives,
    resolve_week_objectives,
)
from validate_page_objectives import (  # noqa: E402
    extract_lo_ids,
    infer_week_from_path,
    validate_page,
)

# Canonical fixture: a four-week course with two "Week 1-2" COs and two
# "Week 3-4" COs plus two terminal objectives. This mirrors the shape of
# ``<COURSE>_objectives.json`` on a smaller scale and is deliberately
# self-contained so the test doesn't depend on gitignored input files.
FIXTURE_OBJECTIVES = {
    "course_title": "Mini LO Specificity Fixture",
    "description": "Small canonical objectives JSON for the LO-specificity tests.",
    "terminal_objectives": [
        {
            "id": "TO-01",
            "statement": "Evaluate content against accessibility standards",
            "bloomLevel": "evaluate",
        },
        {
            "id": "TO-02",
            "statement": "Design accessible interfaces using semantic HTML",
            "bloomLevel": "create",
        },
    ],
    "chapter_objectives": [
        {
            "chapter": "Week 1-2: Foundations",
            "objectives": [
                {"id": "CO-01", "statement": "Explain POUR", "bloomLevel": "understand"},
                {"id": "CO-02", "statement": "Describe disability models", "bloomLevel": "understand"},
            ],
        },
        {
            "chapter": "Week 3-4: Visual Design",
            "objectives": [
                {"id": "CO-03", "statement": "Apply color contrast rules", "bloomLevel": "apply"},
                {"id": "CO-04", "statement": "Implement keyboard navigation", "bloomLevel": "apply"},
            ],
        },
    ],
}


@pytest.fixture
def canonical_path(tmp_path):
    p = tmp_path / "objectives.json"
    p.write_text(json.dumps(FIXTURE_OBJECTIVES))
    return p


@pytest.fixture
def canonical(canonical_path):
    return load_canonical_objectives(canonical_path)


# ---------------------------------------------------------------------------
# Unit tests: resolve_week_objectives
# ---------------------------------------------------------------------------

class TestResolveWeekObjectives:
    """Unit tests for the deterministic LO-selection function."""

    def test_week_3_returns_tos_plus_week_3_chapter_cos(self, canonical):
        """Given week=3, should return all TOs plus COs from the "Week 3-4" chapter."""
        result = resolve_week_objectives(3, canonical)
        ids = [o["id"] for o in result]
        assert ids == ["TO-01", "TO-02", "CO-03", "CO-04"], (
            "Week 3 must receive both terminal objectives and both CO-03/CO-04 "
            "(declared under 'Week 3-4: Visual Design'). "
            f"Got: {ids}"
        )

    def test_week_0_or_unmapped_returns_terminal_objectives_only(self, canonical):
        """Week 0 (course overview / no chapter cover) returns only TOs."""
        result_zero = resolve_week_objectives(0, canonical)
        ids_zero = [o["id"] for o in result_zero]
        assert ids_zero == ["TO-01", "TO-02"], (
            "Week 0 has no chapter objectives; emitter must fall back to terminal "
            f"objectives only. Got: {ids_zero}"
        )

        # Also true for weeks beyond the declared chapter ranges.
        result_far = resolve_week_objectives(99, canonical)
        ids_far = [o["id"] for o in result_far]
        assert ids_far == ["TO-01", "TO-02"], (
            f"Unmapped week must return TOs only. Got: {ids_far}"
        )

    def test_week_1_and_week_2_both_get_same_chapter_cos(self, canonical):
        """A Week 1-2 chapter must apply to BOTH week 1 and week 2 pages."""
        w1 = [o["id"] for o in resolve_week_objectives(1, canonical)]
        w2 = [o["id"] for o in resolve_week_objectives(2, canonical)]
        assert "CO-01" in w1 and "CO-02" in w1
        assert "CO-01" in w2 and "CO-02" in w2
        assert "CO-03" not in w1, (
            "CO-03 belongs to Week 3-4, must not leak into Week 1. Got: " + str(w1)
        )

    def test_objectives_returned_in_generator_format(self, canonical):
        """Returned LO dicts use ``bloom_level`` (snake_case) not ``bloomLevel``."""
        result = resolve_week_objectives(1, canonical)
        for lo in result:
            assert "bloom_level" in lo, f"Missing bloom_level on {lo!r}"
            # bloomLevel should not bleed through from the canonical JSON.
            assert "bloomLevel" not in lo


# ---------------------------------------------------------------------------
# Integration test: generate_week emits canonical IDs
# ---------------------------------------------------------------------------

class TestGenerateWeekCanonicalIDs:
    """End-to-end: generate_week with canonical objectives emits canonical IDs."""

    def test_generated_week_json_ld_uses_canonical_ids(self, tmp_path, canonical):
        """Week 3 pages must carry CO-03/CO-04 plus TO-01/TO-02 — not W03-* IDs."""
        # Minimal week data with an invented week-local ID list (what the
        # content-generator agent would have produced). After the fix,
        # generate_week should override these with canonical LOs.
        week_data = {
            "week_number": 3,
            "title": "Visual Design",
            "objectives": [
                {"id": "W03-CO-01", "statement": "legacy local", "bloom_level": "apply"},
                {"id": "W03-CO-02", "statement": "legacy local", "bloom_level": "apply"},
            ],
            "overview_text": ["Intro paragraph."],
            "content_modules": [],
            "key_takeaways": ["Something."],
        }
        out = tmp_path / "out"
        generate_week(week_data, out, "TEST_COURSE", canonical_objectives=canonical)

        overview = (out / "week_03" / "week_03_overview.html").read_text()
        ids = extract_lo_ids(overview)
        assert ids is not None, "Overview page must include a JSON-LD block"
        # Canonical IDs must be present; week-local IDs must NOT leak in.
        assert set(ids) == {"TO-01", "TO-02", "CO-03", "CO-04"}, (
            f"Emitted JSON-LD must reference canonical IDs for week 3; got {ids}"
        )
        for legacy in ("W03-CO-01", "W03-CO-02"):
            assert legacy not in ids, (
                f"Week-local ID {legacy} must not appear in JSON-LD when "
                f"canonical objectives are supplied; got {ids}"
            )

    def test_generate_week_without_canonical_preserves_legacy_behavior(self, tmp_path):
        """With ``canonical_objectives=None`` the week's own list is emitted as-is."""
        week_data = {
            "week_number": 3,
            "title": "Visual Design",
            "objectives": [
                {"id": "W03-CO-01", "statement": "legacy local", "bloom_level": "apply"},
            ],
            "overview_text": ["Intro."],
            "content_modules": [],
            "key_takeaways": ["k"],
        }
        out = tmp_path / "out"
        generate_week(week_data, out, "TEST_COURSE", canonical_objectives=None)
        overview = (out / "week_03" / "week_03_overview.html").read_text()
        ids = extract_lo_ids(overview)
        assert ids == ["W03-CO-01"], (
            "Legacy callers passing no canonical must keep emitting the "
            f"week_data objectives unchanged. Got: {ids}"
        )


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidator:
    """End-to-end validator tests covering the pass and buggy-regression cases."""

    def _minimal_html(self, lo_ids):
        """Fabricate a minimal HTML page with a JSON-LD block that lists lo_ids."""
        payload = {
            "@context": "https://ed4all.dev/ns/courseforge/v1",
            "@type": "CourseModule",
            "learningObjectives": [
                {"id": lid, "statement": "x", "bloomLevel": "apply"}
                for lid in lo_ids
            ],
        }
        return (
            "<!DOCTYPE html><html><head>"
            "<title>p</title>"
            '<script type="application/ld+json">'
            + json.dumps(payload)
            + "</script></head><body></body></html>"
        )

    def test_validator_passes_when_ids_subset_of_week(self, tmp_path, canonical):
        """A page with canonical, week-appropriate IDs must pass validation."""
        week3_dir = tmp_path / "week_03"
        week3_dir.mkdir()
        page = week3_dir / "week_03_overview.html"
        page.write_text(self._minimal_html(["TO-01", "CO-03", "CO-04"]))

        ok, msg = validate_page(page, canonical)
        assert ok, f"Validator rejected a correct page: {msg}"

    def test_validator_fails_when_full_lo_set_emitted(self, tmp_path, canonical):
        """The buggy pattern (every page emits ALL LOs) must be flagged.

        This is the pre-fix behaviour: a week-3 page tagging itself with
        CO-01/CO-02 (which belong to weeks 1-2) must fail.
        """
        week3_dir = tmp_path / "week_03"
        week3_dir.mkdir()
        page = week3_dir / "week_03_overview.html"
        page.write_text(
            self._minimal_html(["TO-01", "TO-02", "CO-01", "CO-02", "CO-03", "CO-04"])
        )

        ok, msg = validate_page(page, canonical)
        assert not ok, (
            "Validator must reject pages that leak other weeks' LO IDs — "
            "this is the specific regression the fix is defending against."
        )
        assert "CO-01" in msg and "CO-02" in msg, (
            "Failure message should name the offending extraneous IDs "
            f"(expected CO-01/CO-02 to be flagged). Got: {msg}"
        )

    def test_validator_infers_week_from_path(self):
        """Path-based week inference should handle ``week_07`` style paths."""
        assert infer_week_from_path(Path("exports/x/week_07/foo.html")) == 7
        assert infer_week_from_path(Path("week_01_overview.html")) == 1
        assert infer_week_from_path(Path("no_week_here/foo.html")) is None

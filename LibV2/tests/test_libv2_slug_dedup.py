"""Wave 74 cleanup — LibV2 slug dedup regression tests.

Bug observed (2026-04-24): ``python -m Trainforge.process_course
--import-to-libv2`` derived a slug from ``course_code`` and
``course_title`` and produced ``rdf-shacl-550-rdf-shacl-550`` because:

    Courseforge's IMSCC packager writes the manifest title as
    ``f"{course_code}: {course_title}"`` (Courseforge/scripts/
    package_multifile_imscc.py:145), and Trainforge's IMSCC parser
    falls back to ``course_code`` when the manifest carries no other
    usable title (Trainforge/process_course.py:974). So the title round-
    tripped as ``"RDF_SHACL_550: RDF_SHACL_550"`` and the LibV2 importer's
    ``slugify(title)`` doubled the code into the slug.

These tests pin ``derive_course_slug`` so we never regress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LibV2.tools.libv2.importer import derive_course_slug, slugify  # noqa: E402


@pytest.mark.unit
class TestSlugDedup:
    """``derive_course_slug`` must collapse code-prefixed titles."""

    def test_libv2_slug_strips_course_code_prefix(self):
        """Title carrying a ``{code}: `` prefix collapses to ``{code} {rest}``."""
        slug = derive_course_slug(
            course_code="RDF_SHACL_550",
            course_title="RDF_SHACL_550: RDF Course",
        )
        # The code stays as the leading slug segment; the title remainder
        # contributes the rest. No doubled ``rdf-shacl-550-rdf-shacl-550``.
        assert slug == "rdf-shacl-550-rdf-course", slug

    def test_libv2_slug_no_doubling(self):
        """Title equals course_code → use just slugify(course_code)."""
        slug = derive_course_slug(
            course_code="RDF_SHACL_550",
            course_title="RDF_SHACL_550",
        )
        assert slug == "rdf-shacl-550", slug

    def test_libv2_slug_no_doubling_with_code_colon_code(self):
        """Title is ``{code}: {code}`` — the exact today-bug shape."""
        slug = derive_course_slug(
            course_code="RDF_SHACL_550",
            course_title="RDF_SHACL_550: RDF_SHACL_550",
        )
        # Both code-prefixes strip out, title remainder is empty, slug
        # is just slugify(course_code).
        assert slug == "rdf-shacl-550", slug
        # Critical: the bug-shape we are guarding against MUST NOT happen.
        assert slug != "rdf-shacl-550-rdf-shacl-550"

    def test_libv2_slug_unchanged_when_distinct(self):
        """Distinct title → slug includes both code + title."""
        slug = derive_course_slug(
            course_code="MAT_101",
            course_title="College Algebra",
        )
        assert slug == "mat-101-college-algebra", slug

    def test_libv2_slug_handles_no_course_code(self):
        """Legacy callers that pass only a title still work (slugify-only)."""
        slug = derive_course_slug(
            course_code="",
            course_title="Introduction to Physics",
        )
        assert slug == "introduction-to-physics", slug

    def test_libv2_slug_handles_no_title(self):
        """Code-only callers get slugify(code)."""
        slug = derive_course_slug(
            course_code="PHYS_101",
            course_title="",
        )
        assert slug == "phys-101", slug

    def test_libv2_slug_falls_back_when_both_empty(self):
        """When code + title are both empty, fallback (e.g. dir name) wins."""
        slug = derive_course_slug(
            course_code="",
            course_title="",
            fallback="ed4all-mini-course",
        )
        assert slug == "ed4all-mini-course", slug

    def test_libv2_slug_strips_dash_separator(self):
        """``{code} - {title}`` separator is collapsed too."""
        slug = derive_course_slug(
            course_code="BIO_201",
            course_title="BIO_201 - Cellular Biology",
        )
        assert slug == "bio-201-cellular-biology", slug

    def test_libv2_slug_strips_repeated_prefix(self):
        """``{code}: {code}: {title}`` — strip the prefix iteratively."""
        slug = derive_course_slug(
            course_code="CHEM_101",
            course_title="CHEM_101: CHEM_101: Organic Chemistry",
        )
        assert slug == "chem-101-organic-chemistry", slug

    def test_libv2_slug_case_insensitive_prefix_match(self):
        """Prefix match ignores case (manifests aren't case-canonical)."""
        slug = derive_course_slug(
            course_code="MAT_101",
            course_title="mat_101: Linear Algebra",
        )
        assert slug == "mat-101-linear-algebra", slug


@pytest.mark.unit
def test_slugify_alone_is_unchanged():
    """Regression guard — ``slugify`` keeps its old behaviour. The dedup
    happens upstream in ``derive_course_slug``."""
    # The bug-shape input: when slugify is called in isolation on a
    # ``code: code`` title, doubling is the expected (legacy) behaviour.
    # ``derive_course_slug`` is what guards against it.
    assert slugify("RDF_SHACL_550: RDF_SHACL_550") == "rdf-shacl-550-rdf-shacl-550"

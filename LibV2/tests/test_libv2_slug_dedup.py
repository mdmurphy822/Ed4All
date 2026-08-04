"""LibV2 slug-deduplication regression tests.

``python -m Trainforge.pipeline.process_course --import-to-libv2`` derives a slug from
``course_code`` and ``course_title``. The import contract must avoid a doubled
``<code>-<code>`` slug when:

    Courseforge's IMSCC packager writes the manifest title as
    ``f"{course_code}: {course_title}"`` (Courseforge/scripts/
    package_multifile_imscc.py:145), and Trainforge's IMSCC parser
    falls back to ``course_code`` when the manifest carries no other
    usable title (Trainforge/pipeline/process_course.py:974). So the title round-
    tripped as ``"<COURSE_CODE>: <COURSE_CODE>"`` and the LibV2 importer's
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
            course_code="CATALOG_ALPHA",
            course_title="CATALOG_ALPHA: Synthetic Course",
        )
        # The code stays as the leading slug segment; the title remainder
        # contributes the rest. No doubled ``catalog-alpha-catalog-alpha``.
        assert slug == "catalog-alpha-synthetic-course", slug

    def test_libv2_slug_no_doubling(self):
        """Title equals course_code → use just slugify(course_code)."""
        slug = derive_course_slug(
            course_code="CATALOG_ALPHA",
            course_title="CATALOG_ALPHA",
        )
        assert slug == "catalog-alpha", slug

    def test_libv2_slug_no_doubling_with_code_colon_code(self):
        """Title is ``{code}: {code}`` — the exact today-bug shape."""
        slug = derive_course_slug(
            course_code="CATALOG_ALPHA",
            course_title="CATALOG_ALPHA: CATALOG_ALPHA",
        )
        # Both code-prefixes strip out, title remainder is empty, slug
        # is just slugify(course_code).
        assert slug == "catalog-alpha", slug
        # Critical: the bug-shape we are guarding against MUST NOT happen.
        assert slug != "catalog-alpha-catalog-alpha"

    def test_libv2_slug_unchanged_when_distinct(self):
        """Distinct title → slug includes both code + title."""
        slug = derive_course_slug(
            course_code="TST_907",
            course_title="Synthetic Topic Alpha",
        )
        assert slug == "tst-907-synthetic-topic-alpha", slug

    def test_libv2_slug_handles_no_course_code(self):
        """Legacy callers that pass only a title still work (slugify-only)."""
        slug = derive_course_slug(
            course_code="",
            course_title="Synthetic Topic Beta",
        )
        assert slug == "synthetic-topic-beta", slug

    def test_libv2_slug_handles_no_title(self):
        """Code-only callers get slugify(code)."""
        slug = derive_course_slug(
            course_code="TST_910",
            course_title="",
        )
        assert slug == "tst-910", slug

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
            course_code="TST_903",
            course_title="TST_903 - Synthetic Topic Gamma",
        )
        assert slug == "tst-903-synthetic-topic-gamma", slug

    def test_libv2_slug_strips_repeated_prefix(self):
        """``{code}: {code}: {title}`` — strip the prefix iteratively."""
        slug = derive_course_slug(
            course_code="TST_904",
            course_title="TST_904: TST_904: Synthetic Topic Delta",
        )
        assert slug == "tst-904-synthetic-topic-delta", slug

    def test_libv2_slug_case_insensitive_prefix_match(self):
        """Prefix match ignores case (manifests aren't case-canonical)."""
        slug = derive_course_slug(
            course_code="TST_907",
            course_title="tst_907: Synthetic Topic Epsilon",
        )
        assert slug == "tst-907-synthetic-topic-epsilon", slug


@pytest.mark.unit
def test_slugify_alone_is_unchanged():
    """Regression guard — ``slugify`` keeps its old behaviour. The dedup
    happens upstream in ``derive_course_slug``."""
    # The bug-shape input: when slugify is called in isolation on a
    # ``code: code`` title, doubling is the expected (legacy) behaviour.
    # ``derive_course_slug`` is what guards against it.
    assert slugify("CATALOG_ALPHA: CATALOG_ALPHA") == "catalog-alpha-catalog-alpha"

"""Tests for Wave 24 _page_roles_for_week dynamic page allocation.

Before Wave 24, pipeline_tools.py hardcoded a 5-tuple
(overview, content_01, application, self_check, summary) for every week
regardless of LO count. This helper scales the content_NN count with
lo_count while preserving the overview + application + self_check +
summary scaffolding.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools._content_gen_helpers import _page_roles_for_week


def test_single_lo_minimum_three_pages():
    """1 LO → at least 3 pages (overview + content_01 + summary)."""
    roles = _page_roles_for_week(1)
    assert len(roles) >= 3
    assert "overview" in roles
    assert "summary" in roles


def test_two_lo_produces_standard_layout():
    """2 LOs → 5 standard pages (1 content page via ceil(2/2)=1)."""
    roles = _page_roles_for_week(2)
    assert "overview" in roles
    assert "content_01" in roles
    assert "application" in roles
    assert "self_check" in roles
    assert "summary" in roles


def test_four_lo_produces_two_content_pages():
    """4 LOs → 2 content pages via ceil(4/2)=2."""
    roles = _page_roles_for_week(4)
    content_pages = [r for r in roles if r.startswith("content_")]
    assert len(content_pages) == 2
    # Naming is zero-padded 2-digit.
    assert "content_01" in roles
    assert "content_02" in roles


def test_twenty_lo_capped_at_max():
    """20 LOs would yield 10 content pages; total capped at 10 pages."""
    roles = _page_roles_for_week(20)
    assert len(roles) <= 10
    # Tail labels preserved.
    assert roles[-1] == "summary"
    assert "application" in roles
    assert "self_check" in roles
    # At least some content_NN pages.
    assert any(r.startswith("content_") for r in roles)


def test_zero_lo_still_minimal_layout():
    """0 LOs → minimum 3 pages (no crash, no infinite expansion)."""
    roles = _page_roles_for_week(0)
    assert len(roles) >= 3
    assert "overview" in roles


def test_negative_lo_handled():
    """Negative counts treated as zero."""
    roles = _page_roles_for_week(-5)
    assert len(roles) >= 3

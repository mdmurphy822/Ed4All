"""Wave 76 — domain whitelist regression tests.

The pre-Wave-76 ``validate_taxonomy_compliance`` did a strict equality
check against the canonical taxonomy keys (slug form, e.g.
``computer-science``), so manifests that emitted the natural-language
form (``"computer science"``) were marked invalid even though every
ed-tech tool in the pipeline emits the human-readable form. This test
locks in the case-insensitive + alias-aware match introduced in Wave 76.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from LibV2.tools.libv2.validator import (
    _domain_matches,
    validate_taxonomy_compliance,
)


def _build_archive(tmp_path: Path, primary_domain: str, division: str = "STEM") -> Path:
    """Write a minimal LibV2 course directory with classification."""
    course_dir = tmp_path / "course"
    course_dir.mkdir()
    manifest = {
        "libv2_version": "1.2.0",
        "slug": "test-course",
        "classification": {
            "division": division,
            "primary_domain": primary_domain,
            "secondary_domains": [],
            "subdomains": [],
            "topics": [],
            "subtopics": [],
        },
    }
    (course_dir / "manifest.json").write_text(json.dumps(manifest))
    return course_dir


@pytest.fixture
def repo_root() -> Path:
    """Real project root so the validator finds the canonical taxonomy."""
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Pure helper: _domain_matches
# ---------------------------------------------------------------------------


def test_domain_matches_slug_to_slug() -> None:
    assert _domain_matches("computer-science", "computer-science")


def test_domain_matches_space_to_slug() -> None:
    """The bug case — manifest emits ``"computer science"`` (space form)
    against canonical slug ``computer-science``. Must resolve."""
    assert _domain_matches("computer science", "computer-science")


def test_domain_matches_titlecase_to_slug() -> None:
    assert _domain_matches("Computer Science", "computer-science")


def test_domain_matches_alias_software_engineering() -> None:
    """Software engineering is a recognized CS subdomain alias."""
    assert _domain_matches("Software Engineering", "computer-science")
    assert _domain_matches("software engineering", "computer-science")


def test_domain_matches_alias_information_systems() -> None:
    assert _domain_matches("Information Systems", "computer-science")


def test_domain_matches_unknown_domain_rejected() -> None:
    assert not _domain_matches("astrology", "computer-science")
    assert not _domain_matches("astrology", "physics")


def test_domain_matches_empty_string_rejected() -> None:
    assert not _domain_matches("", "computer-science")


def test_domain_matches_other_canonical_pairs() -> None:
    """Spot-check that the slug↔space rule applies broadly, not just CS."""
    assert _domain_matches("data science", "data-science")
    assert _domain_matches("environmental science", "environmental-science")
    assert _domain_matches("educational technology", "educational-technology")


# ---------------------------------------------------------------------------
# End-to-end: validate_taxonomy_compliance against an archive on disk
# ---------------------------------------------------------------------------


def test_validate_accepts_computer_science_space_form(tmp_path: Path, repo_root: Path) -> None:
    """The exact failure mode reported on rdf-shacl-550: manifest's
    ``primary_domain`` is ``"computer science"`` (lowercase + space)."""
    archive = _build_archive(tmp_path, "computer science")
    result = validate_taxonomy_compliance(archive, repo_root)
    assert result.valid, f"validation should pass; errors: {result.errors}"
    assert not result.errors


def test_validate_accepts_titlecase_computer_science(tmp_path: Path, repo_root: Path) -> None:
    archive = _build_archive(tmp_path, "Computer Science")
    result = validate_taxonomy_compliance(archive, repo_root)
    assert result.valid
    assert not result.errors


def test_validate_accepts_canonical_slug_form(tmp_path: Path, repo_root: Path) -> None:
    """Direct slug form is the legacy contract — must keep working."""
    archive = _build_archive(tmp_path, "computer-science")
    result = validate_taxonomy_compliance(archive, repo_root)
    assert result.valid


def test_validate_rejects_unknown_domain(tmp_path: Path, repo_root: Path) -> None:
    """Genuine bad inputs must still fail closed."""
    archive = _build_archive(tmp_path, "Astrology")
    result = validate_taxonomy_compliance(archive, repo_root)
    assert not result.valid
    assert any("Unknown domain" in e for e in result.errors)


def test_validate_rejects_unknown_division(tmp_path: Path, repo_root: Path) -> None:
    """Division enum is still strict (STEM / ARTS) — no aliasing there."""
    archive = _build_archive(tmp_path, "computer science", division="UNKNOWN")
    result = validate_taxonomy_compliance(archive, repo_root)
    assert not result.valid

"""Tests for ``gui.services.answer_service`` engine resolution (Workstream 0).

Pins the ``auto`` engine choice to the benchmark-selected default: ``auto``
resolves to ``hybrid-rrf`` when a vector index manifest exists for the course,
else falls back to ``lexical``. An explicit engine always passes through
verbatim (anti-silent-degradation). No fastapi / model / network needed — the
resolver is a cheap filesystem stat over a tmp LibV2 layout.
"""

from __future__ import annotations

from pathlib import Path

from gui.services import answer_service


def _make_course(tmp_path: Path, slug: str, *, with_index: bool) -> Path:
    """Lay out a tmp LibV2 course; optionally drop a vector-index manifest."""
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / slug
    course_dir.mkdir(parents=True)
    if with_index:
        vidx = course_dir / "vector_index"
        vidx.mkdir()
        (vidx / "manifest.json").write_text("{}", encoding="utf-8")
    return libv2_root


def test_auto_resolves_hybrid_rrf_with_index(tmp_path: Path):
    slug = "course-with-index"
    libv2_root = _make_course(tmp_path, slug, with_index=True)
    assert answer_service._resolve_engine("auto", libv2_root, slug) == "hybrid-rrf"


def test_auto_resolves_lexical_without_index(tmp_path: Path):
    slug = "course-no-index"
    libv2_root = _make_course(tmp_path, slug, with_index=False)
    assert answer_service._resolve_engine("auto", libv2_root, slug) == "lexical"


def test_explicit_engine_passes_through_verbatim(tmp_path: Path):
    slug = "course-no-index"
    libv2_root = _make_course(tmp_path, slug, with_index=False)
    # An explicit semantic against a missing index is NOT downgraded here — the
    # router surfaces the typed 503; the resolver returns it verbatim.
    assert answer_service._resolve_engine("semantic", libv2_root, slug) == "semantic"
    assert (
        answer_service._resolve_engine("hybrid-rrf", libv2_root, slug)
        == "hybrid-rrf"
    )
    assert answer_service._resolve_engine("lexical", libv2_root, slug) == "lexical"


def test_default_empty_engine_treated_as_auto(tmp_path: Path):
    slug = "course-with-index"
    libv2_root = _make_course(tmp_path, slug, with_index=True)
    # An empty / None engine defaults to auto → hybrid-rrf with an index.
    assert answer_service._resolve_engine("", libv2_root, slug) == "hybrid-rrf"

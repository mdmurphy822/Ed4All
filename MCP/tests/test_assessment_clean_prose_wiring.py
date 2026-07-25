"""Wiring net for the assessment clean-prose mining view.

``lib/assessment/tests/test_source_prose.py`` pins the filter's behaviour.
This file pins the part that has burned us before: that the filter is reached
by the phase that is supposed to use it, and that turning it on actually
changes what the generator would mine. A guard validated only in isolation can
ship inert if the phase routes around it.

Everything here is synthetic — no course slug, no corpus fixture.
"""

from pathlib import Path

import pytest

pytest.importorskip("lxml")

from MCP.tools.pipeline_tools import _resolve_staged_html_dir  # noqa: E402
from lib.assessment.source_prose import (  # noqa: E402
    build_prose_filter,
    clean_chunks,
    resolve_clean_prose,
    resolve_source_html_paths,
)

ALT = "A shaded circle beside a bar, indicating the completed portion."
PROSE = "A ratio compares two quantities that share the same unit of measure."

DOC = f"""<html><body><main>
  <section data-semantik-block-role="figure"><img alt="{ALT}"/></section>
  <section><p>{PROSE}</p></section>
</main></body></html>"""


def _staging(tmp_path: Path, slug: str, stamp: str) -> Path:
    """Build the ``exports/../inputs/textbooks/TTC_<slug>_<stamp>`` layout."""
    root = tmp_path / "Courseforge"
    staged = root / "inputs" / "textbooks" / f"TTC_{slug}_{stamp}"
    staged.mkdir(parents=True)
    (root / "exports").mkdir(parents=True, exist_ok=True)
    return staged


def test_resolve_staged_html_dir_finds_the_course_staging(tmp_path):
    slug = "synthetic-course"
    staged = _staging(tmp_path, slug, "20200101_000000")
    project = tmp_path / "Courseforge" / "exports" / "PROJ-synthetic-course-1"
    project.mkdir()
    assert _resolve_staged_html_dir(project, slug) == staged


def test_resolve_staged_html_dir_prefers_the_newest(tmp_path):
    slug = "synthetic-course"
    old = _staging(tmp_path, slug, "20200101_000000")
    new = _staging(tmp_path, slug, "20200202_000000")
    import os
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    project = tmp_path / "Courseforge" / "exports" / "PROJ-synthetic-course-1"
    project.mkdir()
    assert _resolve_staged_html_dir(project, slug) == new


def test_resolve_staged_html_dir_returns_none_when_absent(tmp_path):
    project = tmp_path / "exports" / "PROJ-nothing-1"
    project.mkdir(parents=True)
    assert _resolve_staged_html_dir(project, "no-such-course") is None


def test_flag_off_leaves_the_mining_pool_untouched(tmp_path, monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSESSMENT_CLEAN_PROSE", raising=False)
    assert resolve_clean_prose() is False


def test_flag_on_removes_apparatus_from_the_mining_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSESSMENT_CLEAN_PROSE", "1")
    slug = "synthetic-course"
    staged = _staging(tmp_path, slug, "20200101_000000")
    (staged / "unit_one_accessible.html").write_text(DOC, encoding="utf-8")
    project = tmp_path / "Courseforge" / "exports" / "PROJ-synthetic-course-1"
    project.mkdir()

    chunks = [{
        "id": "chunk_1",
        "text": f"{PROSE} {ALT}",
        "source": {"item_path": "unit_one_accessible.html"},
    }]

    assert resolve_clean_prose() is True
    html_dir = _resolve_staged_html_dir(project, slug)
    assert html_dir == staged
    paths = resolve_source_html_paths(chunks, [html_dir])
    assert paths == [staged / "unit_one_accessible.html"]
    filt = build_prose_filter(paths)
    cleaned, stats = clean_chunks(chunks, filt)

    assert ALT not in cleaned[0]["text"], "apparatus survived the mining view"
    assert PROSE.rstrip(".") in cleaned[0]["text"], "prose was over-pruned"
    assert stats["changed"] == 1


def test_chunkset_fingerprint_moves_when_the_flag_flips(tmp_path):
    """Enabling the filter must re-roll cached units, not replay them.

    ``_assessment_chunkset_sha_val`` is computed AFTER the clean-prose pass
    precisely so a mid-build flip invalidates the per-unit resume sidecars.
    """
    from MCP.tools.pipeline_tools import _assessment_chunkset_sha

    slug = "synthetic-course"
    staged = _staging(tmp_path, slug, "20200101_000000")
    (staged / "unit_one_accessible.html").write_text(DOC, encoding="utf-8")
    chunks = [{
        "id": "chunk_1",
        "text": f"{PROSE} {ALT}",
        "source": {"item_path": "unit_one_accessible.html"},
    }]
    before = _assessment_chunkset_sha(chunks)
    filt = build_prose_filter([staged / "unit_one_accessible.html"])
    cleaned, _ = clean_chunks(chunks, filt)
    after = _assessment_chunkset_sha(cleaned)
    assert before != after

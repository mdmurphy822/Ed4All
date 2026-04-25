"""Wave 75 — LibV2 importer must copy objectives.json into the archive.

ChatGPT's review of the RDF_SHACL_550 archive flagged that
``course.json`` declared only the 7 terminal outcomes — the 29
component objectives synthesized by Courseforge never propagated
into the LibV2 archive, so 312 chunk ``learning_outcome_refs`` to
``co-*`` codes couldn't resolve.

Wave 75 fixes the emit + import sides:

  * Trainforge writes a Wave-75 ``objectives.json`` sidecar carrying
    the full TO-/CO- hierarchy.
  * LibV2's ``import_course`` copies that sidecar into the archive
    root next to ``course.json``.

These tests pin the importer copy step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LibV2.tools.libv2.importer import import_course  # noqa: E402


def _write_minimal_sourceforge_dir(
    src_dir: Path,
    *,
    course_code: str = "WAVE75_TEST",
    course_title: str = "Wave 75 Test Course",
    include_objectives: bool = True,
) -> None:
    """Build a minimal Trainforge output dir an import_course call accepts."""
    src_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "sourceforge_version": "test-fixture",
        "export_timestamp": "2026-04-24T00:00:00Z",
        "course_id": course_code,
        "course_title": course_title,
        "title": course_title,
        "statistics": {"chunks": 0, "concepts": 0},
    }
    (src_dir / "manifest.json").write_text(json.dumps(manifest))

    course_data = {
        "course_code": course_code,
        "title": course_title,
        "learning_outcomes": [
            {
                "id": "to-01",
                "statement": "Demonstrate the basics.",
                "hierarchy_level": "terminal",
                "bloom_level": "understand",
            },
            {
                "id": "co-01",
                "statement": "Identify foundational terms.",
                "hierarchy_level": "chapter",
                "type": "component",
                "bloom_level": "remember",
            },
        ],
    }
    (src_dir / "course.json").write_text(json.dumps(course_data))

    if include_objectives:
        objectives_data = {
            "schema_version": "v1",
            "course_code": course_code,
            "terminal_outcomes": [
                {
                    "id": "to-01",
                    "statement": "Demonstrate the basics.",
                    "bloom_level": "understand",
                }
            ],
            "component_objectives": [
                {
                    "id": "co-01",
                    "parent_terminal": "to-01",
                    "statement": "Identify foundational terms.",
                    "bloom_level": "remember",
                    "week": 1,
                }
            ],
            "objective_count": {"terminal": 1, "component": 1},
        }
        (src_dir / "objectives.json").write_text(json.dumps(objectives_data))

    # Required sub-trees the importer expects to copy.
    for sub in ["corpus", "graph", "pedagogy", "training_specs", "quality"]:
        (src_dir / sub).mkdir(parents=True, exist_ok=True)
    # Empty chunks.jsonl satisfies LibV2's chunk-count expectations.
    (src_dir / "corpus" / "chunks.jsonl").write_text("")
    (src_dir / "corpus" / "corpus_stats.json").write_text(json.dumps({}))
    (src_dir / "training_specs" / "dataset_config.json").write_text(
        json.dumps({"statistics": {"total_tokens": 0}})
    )


def _write_minimal_libv2_root(repo_root: Path) -> None:
    (repo_root / "courses").mkdir(parents=True, exist_ok=True)
    (repo_root / "catalog").mkdir(parents=True, exist_ok=True)


@pytest.mark.unit
def test_importer_copies_objectives_json_into_archive(tmp_path):
    """When the source dir carries objectives.json, the archive does too."""
    src = tmp_path / "src"
    repo = tmp_path / "libv2"
    _write_minimal_sourceforge_dir(src)
    _write_minimal_libv2_root(repo)

    slug = import_course(
        source_dir=src,
        repo_root=repo,
        division="STEM",
        domain="computer-science",
        strict_validation=False,
    )

    archive = repo / "courses" / slug
    assert archive.exists(), f"archive not created: {archive}"
    assert (archive / "course.json").exists()
    objectives_path = archive / "objectives.json"
    assert objectives_path.exists(), (
        "Wave 75 regression: objectives.json must land in the LibV2 archive"
    )

    data = json.loads(objectives_path.read_text())
    assert data["schema_version"] == "v1"
    assert data["course_code"] == "WAVE75_TEST"
    assert data["objective_count"] == {"terminal": 1, "component": 1}
    assert any(co["id"] == "co-01" for co in data["component_objectives"])
    assert any(to["id"] == "to-01" for to in data["terminal_outcomes"])


@pytest.mark.unit
def test_importer_skips_objectives_when_source_lacks_it(tmp_path):
    """Pre-Wave-75 source dirs without objectives.json are still importable."""
    src = tmp_path / "src"
    repo = tmp_path / "libv2"
    _write_minimal_sourceforge_dir(src, include_objectives=False)
    _write_minimal_libv2_root(repo)

    slug = import_course(
        source_dir=src,
        repo_root=repo,
        division="STEM",
        domain="computer-science",
        strict_validation=False,
    )

    archive = repo / "courses" / slug
    assert (archive / "course.json").exists()
    assert not (archive / "objectives.json").exists(), (
        "objectives.json must be absent when not present at source — "
        "Wave 75 importer copies optionally, no fabrication."
    )

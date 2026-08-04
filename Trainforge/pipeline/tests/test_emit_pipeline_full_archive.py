"""Fresh hermetic Trainforge emit and packet-validation contract.

This test exercises the public processing path end-to-end:

1. Build a representative IMSCC and objectives sidecar under ``tmp_path`` and
   run ``CourseProcessor.process()`` into a second temporary directory.
2. Mirror the emitted artifacts into the archive layout consumed by LibV2.
3. Run the packet validator
   (:class:`lib.validators.libv2_packet_integrity.PacketIntegrityValidator`)
   in default mode.

The fresh emit must contain a classified concept graph and a non-stub pedagogy
graph with meaningful nodes and typed relationships.

The fixture is small enough for the default suite and never discovers or reads
operator archives.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.pipeline.tests.test_emit_pipeline_enrichment import (  # noqa: E402
    _imscc_manifest,
    _objectives_payload,
    _page_one,
    _page_two,
)


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Build a neutral two-page cartridge and complete objective inventory."""
    imscc_path = tmp_path / "synthetic_systems.imscc"
    with zipfile.ZipFile(imscc_path, "w") as archive:
        archive.writestr("imsmanifest.xml", _imscc_manifest())
        archive.writestr("week_01/content.html", _page_one())
        archive.writestr("week_02/content.html", _page_two())

    objectives = _objectives_payload()
    objectives_path = tmp_path / "objectives.json"
    objectives_path.write_text(json.dumps(objectives), encoding="utf-8")
    return imscc_path, objectives_path, objectives


def test_fresh_emit_validates_against_packet_validator(tmp_path):
    """A fresh emit has a non-stub graph and no critical packet issues."""
    from lib.validators.libv2.packet_integrity import PacketIntegrityValidator
    from Trainforge.pipeline.process_course import CourseProcessor

    out = tmp_path / "trainforge_out"
    out.mkdir()
    imscc_path, objectives_path, objectives = _build_fixture(tmp_path)

    proc = CourseProcessor(
        imscc_path=str(imscc_path),
        output_dir=str(out),
        course_code=objectives["course_code"],
        domain=objectives["domain"],
        objectives_path=str(objectives_path),
    )
    proc.process()

    # A fresh pedagogy graph must contain useful structure.
    pg = json.loads(
        (out / "graph" / "pedagogy_graph.json").read_text(encoding="utf-8")
    )
    nodes = pg.get("nodes") or []
    edges = pg.get("edges") or []
    relation_types = {e.get("relation_type") for e in edges}

    # Five supplied objectives plus the typed support nodes and emitted chunks
    # produce a nontrivial node and edge inventory.
    assert len(nodes) >= 14, (
        f"pedagogy_graph is too small ({len(nodes)} nodes); expected at least 14."
    )
    assert len(edges) >= 15, (
        f"pedagogy_graph is too sparse ({len(edges)} edges); expected at least 15."
    )
    required_relations = {
        "derived_from_objective",
        "chunk_at_difficulty",
        "supports_outcome",
        "at_bloom_level",
    }
    assert required_relations <= relation_types, (
        "representative emit omitted required "
        f"relations {sorted(required_relations - relation_types)}. "
        f"Got: {sorted(relation_types)}"
    )

    # Mirror the fresh emit into the archive structure expected by validation.
    libv2_archive = tmp_path / "libv2_archive"
    libv2_archive.mkdir()

    # Mirror the Trainforge emit into the LibV2 archive shape.
    import shutil

    for sub in (
        "imscc_chunks", "corpus", "graph", "quality", "training_specs", "pedagogy",
    ):
        src = out / sub
        if src.exists():
            shutil.copytree(src, libv2_archive / sub)
    for f in ("manifest.json", "course.json", "objectives.json"):
        src = out / f
        if src.exists():
            shutil.copy2(src, libv2_archive / f)

    # Default validation mode permits advisory coverage and typing warnings,
    # while structural failures remain critical.
    validator = PacketIntegrityValidator()
    result = validator.validate(libv2_archive)
    assert result.critical_count == 0, (
        f"fresh emit failed packet validator with "
        f"{result.critical_count} critical issues. Issues: "
        + "\n".join(
            f"  [{i.severity}] {i.rule}: {i.message}"
            for i in result.issues
            if i.severity == "critical"
        )
    )

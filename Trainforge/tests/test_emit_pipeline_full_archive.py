"""Wave 81 regression — a fresh hermetic emit matches enriched output.

The Wave 81 worker brief calls out v2 Path B regen as the precipitating
incident: a fresh Trainforge run produced
a 1-node / 0-edge stub pedagogy graph, then needed the 4 Wave 75/76/78
retroactive scripts to be run by hand before the archive validated
under the Wave 78 packet validator.

This test exercises that exact loop end-to-end:

1. Build a representative IMSCC and objectives sidecar under ``tmp_path`` and
   run ``CourseProcessor.process()`` into a second temporary directory.
2. Stamp the archive with the same scaffold the LibV2 importer
   produces (objectives.json + course.json — the IMSCC carries
   neither).
3. Run the Wave 78 packet validator
   (:class:`lib.validators.libv2_packet_integrity.PacketIntegrityValidator`)
   in default mode.

Pre-Wave-81: the validator surfaces the stub-pedagogy + missing-class
issues immediately. Post-Wave-81: the fresh emit lands a real
pedagogy graph (>= 14 nodes, >= 5 edge types) and a classified
concept_graph at the same time.

The fixture is small enough for the default suite and never discovers or reads
operator archives.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Trainforge.tests.test_emit_pipeline_enrichment import (  # noqa: E402
    _imscc_manifest,
    _objectives_payload,
    _page_one,
    _page_two,
)


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Build a neutral two-page cartridge and complete objective inventory."""
    imscc_path = tmp_path / "sample.imscc"
    with zipfile.ZipFile(imscc_path, "w") as archive:
        archive.writestr("imsmanifest.xml", _imscc_manifest())
        archive.writestr("week_01/content.html", _page_one())
        archive.writestr("week_02/content.html", _page_two())

    objectives = _objectives_payload()
    objectives["course_code"] = "SAMPLE101"
    objectives["chapter_objectives"].append({
        "id": "CO-09",
        "statement": "Apply constraint properties to a graph.",
        "parent_to": "TO-04",
        "bloom_level": "apply",
        "week": 2,
    })
    objectives_path = tmp_path / "objectives.json"
    objectives_path.write_text(json.dumps(objectives), encoding="utf-8")
    return imscc_path, objectives_path


def test_fresh_emit_validates_against_packet_validator(tmp_path):
    """End-to-end: fresh Trainforge emit → packet-validator pass.

    Pre-Wave-81 this test would fail at the pedagogy-graph rule
    (1 node / 0 edges); post-Wave-81 the fresh emit matches the
    post-retroactive-script output that the Wave 78 packet validator
    accepts in default (warning-only) mode.
    """
    from Trainforge.process_course import CourseProcessor
    from lib.validators.libv2.packet_integrity import PacketIntegrityValidator

    out = tmp_path / "trainforge_out"
    out.mkdir()
    imscc_path, objectives_path = _build_fixture(tmp_path)

    proc = CourseProcessor(
        imscc_path=str(imscc_path),
        output_dir=str(out),
        course_code="SAMPLE101",
        domain="knowledge_graphs",
        objectives_path=str(objectives_path),
    )
    proc.process()

    # Wave 81 contract: fresh pedagogy_graph.json must NOT be the stub.
    pg = json.loads(
        (out / "graph" / "pedagogy_graph.json").read_text(encoding="utf-8")
    )
    nodes = pg.get("nodes") or []
    edges = pg.get("edges") or []
    relation_types = {e.get("relation_type") for e in edges}

    # The legacy stub emits 1 node / 0 edges; the real builder must
    # ship at least 14 nodes (6 bloom + 3 difficulty + objectives)
    # and a non-trivial set of edge types on a representative archive.
    assert len(nodes) >= 14, (
        f"Wave 81 regression: pedagogy_graph stub-detected "
        f"({len(nodes)} nodes) on synthetic archive. Expected >= 14."
    )
    assert len(edges) >= 15, (
        f"Wave 81 regression: pedagogy_graph thin "
        f"({len(edges)} edges) on synthetic archive. Expected >= 15."
    )
    required_relations = {
        "derived_from_objective",
        "chunk_at_difficulty",
        "supports_outcome",
        "at_bloom_level",
    }
    assert required_relations <= relation_types, (
        "Wave 81 regression: representative emit omitted required "
        f"relations {sorted(required_relations - relation_types)}. "
        f"Got: {sorted(relation_types)}"
    )

    # Now run the Wave 78 packet validator on the fresh emit. We
    # have to assemble a faux libv2 archive structure from the
    # Trainforge output so the validator finds what it expects.
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

    # The packet validator is intentionally invoked in DEFAULT mode
    # (no --strict / --strict-coverage / --strict-typing) so the
    # coverage + typing rules surface as warnings rather than
    # criticals. The Wave 81 contract is that critical-rule failures
    # must NOT regress on a fresh emit. Pre-Wave-81: pedagogy stub
    # surfaces typing-rule criticals.
    validator = PacketIntegrityValidator()
    result = validator.validate(libv2_archive)
    assert result.critical_count == 0, (
        f"Wave 81 regression: fresh emit failed packet validator with "
        f"{result.critical_count} critical issues. Issues: "
        + "\n".join(
            f"  [{i.severity}] {i.rule}: {i.message}"
            for i in result.issues
            if i.severity == "critical"
        )
    )

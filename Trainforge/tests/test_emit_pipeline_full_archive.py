"""Wave 81 regression — fresh emit on rdf-shacl-551-2 matches retroactive output.

The Wave 81 worker brief calls out v2 Path B regen as the precipitating
incident: a fresh Trainforge run on the rdf-shacl-551 IMSCC produced
a 1-node / 0-edge stub pedagogy graph, then needed the 4 Wave 75/76/78
retroactive scripts to be run by hand before the archive validated
under the Wave 78 packet validator.

This test exercises that exact loop end-to-end:

1. Run ``CourseProcessor.process()`` on
   ``LibV2/courses/rdf-shacl-551-2/source/imscc/RDF_SHACL_551.imscc``
   into a tempdir.
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

This test is gated behind ``ED4ALL_RUN_FULL_ARCHIVE_TEST=true`` (or
``--run-full-archive`` pytest opt-in via the ``slow`` marker)
because:

* Running ``CourseProcessor.process()`` on the full IMSCC takes
  ~30 s of wall clock — too slow for the default suite.
* The fixture path depends on the rdf-shacl-551-2 archive being
  present locally; CI runners that don't pull the full LibV2
  shouldn't fail spuriously.

The fast suite at ``test_emit_pipeline_enrichment.py`` covers the
synthetic-fixture contract; this file is the regression bedrock.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Path to the IMSCC source — used as both the fixture probe and the
# pytest skip condition.
ARCHIVE_SLUG = "rdf-shacl-551-2"
ARCHIVE_ROOT = PROJECT_ROOT / "LibV2" / "courses" / ARCHIVE_SLUG
IMSCC_PATH = ARCHIVE_ROOT / "source" / "imscc" / "RDF_SHACL_551.imscc"
SOURCE_OBJECTIVES = ARCHIVE_ROOT / "objectives.json"


def _gated() -> bool:
    """Return True iff the full-archive regression should run.

    Gates: ``ED4ALL_RUN_FULL_ARCHIVE_TEST=true`` env var OR the
    ``slow`` pytest marker passed in via ``-m slow``. We test for the
    env var here (the marker is honored by pytest's selection).
    """
    raw = os.environ.get("ED4ALL_RUN_FULL_ARCHIVE_TEST", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _gated(),
        reason=(
            "full-archive regression skipped by default. Set "
            "ED4ALL_RUN_FULL_ARCHIVE_TEST=true to enable."
        ),
    ),
    pytest.mark.skipif(
        not IMSCC_PATH.exists(),
        reason=f"fixture archive not present: {IMSCC_PATH}",
    ),
]


def test_fresh_emit_validates_against_packet_validator(tmp_path):
    """End-to-end: fresh Trainforge emit → packet-validator pass.

    Pre-Wave-81 this test would fail at the pedagogy-graph rule
    (1 node / 0 edges); post-Wave-81 the fresh emit matches the
    post-retroactive-script output that the Wave 78 packet validator
    accepts in default (warning-only) mode.
    """
    from Trainforge.process_course import CourseProcessor
    from lib.validators.libv2_packet_integrity import PacketIntegrityValidator

    out = tmp_path / "trainforge_out"
    out.mkdir()

    # Use the canonical objectives.json shipped with the archive so
    # the pedagogy builder lands real Outcome / ComponentObjective
    # nodes. (The fresh IMSCC carries no objectives sidecar by
    # design — Trainforge consumes a pre-synthesized one.)
    objectives_path = SOURCE_OBJECTIVES if SOURCE_OBJECTIVES.exists() else None

    proc = CourseProcessor(
        imscc_path=str(IMSCC_PATH),
        output_dir=str(out),
        course_code="RDF_SHACL_551",
        domain="knowledge_graphs",
        objectives_path=str(objectives_path) if objectives_path else None,
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
    # and a non-trivial set of edge types on a real archive.
    assert len(nodes) >= 14, (
        f"Wave 81 regression: pedagogy_graph stub-detected "
        f"({len(nodes)} nodes) on real archive. Expected >= 14."
    )
    assert len(edges) >= 50, (
        f"Wave 81 regression: pedagogy_graph thin "
        f"({len(edges)} edges) on real archive. Expected >= 50."
    )
    # The Wave 78 relation set is the bar: derived_from_objective +
    # concept_supports_outcome + assessment_validates_outcome +
    # chunk_at_difficulty. At least 3 of those 4 must fire on this
    # archive (concept_supports_outcome may not fire if no concept
    # nodes survive classification, but the others always do for an
    # archive with chunks + objectives).
    wave78 = {
        "derived_from_objective",
        "concept_supports_outcome",
        "assessment_validates_outcome",
        "chunk_at_difficulty",
    }
    overlap = wave78 & relation_types
    assert len(overlap) >= 3, (
        f"Wave 81 regression: only {len(overlap)} of the 4 Wave 78 "
        f"relation types fired on a fresh emit. Got: {sorted(overlap)}"
    )

    # Now run the Wave 78 packet validator on the fresh emit. We
    # have to assemble a faux libv2 archive structure from the
    # Trainforge output so the validator finds what it expects.
    libv2_archive = tmp_path / "libv2_archive"
    libv2_archive.mkdir()

    # Mirror the Trainforge emit into the LibV2 archive shape.
    import shutil

    for sub in ("corpus", "graph", "quality", "training_specs", "pedagogy"):
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

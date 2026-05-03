"""Phase 6 ST 20 / Phase 7c ST 17 — concept_graph_sha256 manifest extension tests.

Covers the Phase 6 ST 19 extension to ``LibV2ManifestValidator`` that
recognizes the new ``concept_graph_sha256`` manifest field. Phase 6
landed the gate at warning severity (advisory only); **Phase 7c ST 17
promoted it to critical** alongside the new ``dart_chunks_sha256`` +
``imscc_chunks_sha256`` checks.

Tests (post Phase 7c promotion):
  - Manifest with a valid hash + matching on-disk graph passes
    (no concept_graph issues fire).
  - Manifest missing the hash AND graph file absent — no concept_graph
    issue (legacy / DART-only run).
  - Manifest missing the hash but graph file present — critical fires
    (``MISSING_CONCEPT_GRAPH_SHA256``); blocks.
  - Manifest with a malformed hash (non-hex / wrong length) —
    critical fires (``INVALID_CONCEPT_GRAPH_SHA256``); blocks.
  - Manifest hash divergent from on-disk graph — critical fires
    (``CONCEPT_GRAPH_HASH_MISMATCH``); blocks.
  - Schema regex round-trip: the validator's regex matches the
    canonical 64-hex pattern declared in
    ``schemas/library/course_manifest.schema.json``.

Fixtures include stub ``dart_chunks_sha256`` + ``imscc_chunks_sha256``
fields (now required by Phase 7c ST 17) so the new ``MISSING_*`` checks
don't fire spuriously and pollute the concept_graph-scoped assertions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from lib.validators.libv2_manifest import (
    LibV2ManifestValidator,
    _CONCEPT_GRAPH_SHA256_RE,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def archive_with_graph(tmp_path: Path):
    """Build a minimal LibV2 archive that includes a concept graph file."""
    slug = "phase6-cg-test"
    course_dir = tmp_path / "courses" / slug
    course_dir.mkdir(parents=True)

    # Scaffold dirs (silences MISSING_SCAFFOLD_SUBDIR noise).
    for sub in (
        "corpus", "graph", "training_specs", "quality",
        "source/pdf", "source/html", "source/imscc",
        "concept_graph",
    ):
        (course_dir / sub).mkdir(parents=True, exist_ok=True)

    # Minimal pedagogy + course.json so warning checks don't dominate.
    (course_dir / "pedagogy").mkdir(exist_ok=True)
    (course_dir / "pedagogy" / "model.json").write_text("{}", encoding="utf-8")
    (course_dir / "graph" / "nodes.json").write_text("[]", encoding="utf-8")
    (course_dir / "course.json").write_text(
        json.dumps({"slug": slug, "learning_outcomes": []}),
        encoding="utf-8",
    )

    # Concept graph file with deterministic bytes so hash is stable.
    graph_payload = {
        "kind": "pedagogy",
        "course_id": "PHASE6-CG-TEST",
        "nodes": [],
        "edges": [],
    }
    graph_bytes = json.dumps(graph_payload, indent=2).encode("utf-8")
    graph_file = course_dir / "concept_graph" / "concept_graph_semantic.json"
    graph_file.write_bytes(graph_bytes)
    graph_hash = _sha256(graph_bytes)

    # Source artifact (so source_artifacts integrity check has work).
    pdf_bytes = b"%PDF-1.4 phase6 concept graph fixture" * 5
    pdf_path = course_dir / "source" / "pdf" / "fixture.pdf"
    pdf_path.write_bytes(pdf_bytes)

    manifest: Dict[str, Any] = {
        "libv2_version": "1.2.0",
        "slug": slug,
        "import_timestamp": "2026-05-03T00:00:00.000000",
        "classification": {
            "division": "STEM",
            "primary_domain": "general",
            "subdomains": [],
        },
        "source_artifacts": {
            "pdf": [{
                "path": str(pdf_path),
                "checksum": _sha256(pdf_bytes),
                "size": len(pdf_bytes),
            }],
        },
        "provenance": {
            "source_type": "textbook_to_course_pipeline",
            "import_pipeline_version": "1.0.0",
        },
        "features": {
            "source_provenance": True,
            "evidence_source_provenance": True,
        },
        "concept_graph_sha256": graph_hash,
        # Phase 7c ST 17: dart/imscc chunkset hashes are now required;
        # stub with synthetic 64-hex values so the new MISSING_*
        # checks don't fire and pollute the concept_graph assertions.
        "dart_chunks_sha256": "d" * 64,
        "imscc_chunks_sha256": "1" * 64,
    }
    manifest_path = course_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, course_dir, graph_hash, graph_file


@pytest.fixture
def archive_without_graph(tmp_path: Path):
    """Build a minimal LibV2 archive that has NO concept graph at all."""
    slug = "legacy-no-cg"
    course_dir = tmp_path / "courses" / slug
    course_dir.mkdir(parents=True)
    for sub in (
        "corpus", "graph", "training_specs", "quality",
        "source/pdf", "source/html", "source/imscc",
    ):
        (course_dir / sub).mkdir(parents=True, exist_ok=True)
    (course_dir / "pedagogy").mkdir(exist_ok=True)
    (course_dir / "pedagogy" / "model.json").write_text("{}", encoding="utf-8")
    (course_dir / "graph" / "nodes.json").write_text("[]", encoding="utf-8")
    (course_dir / "course.json").write_text(
        json.dumps({"slug": slug}), encoding="utf-8",
    )
    pdf_bytes = b"%PDF-1.4 no concept graph" * 4
    pdf_path = course_dir / "source" / "pdf" / "fixture.pdf"
    pdf_path.write_bytes(pdf_bytes)

    manifest: Dict[str, Any] = {
        "libv2_version": "1.2.0",
        "slug": slug,
        "import_timestamp": "2026-05-03T00:00:00.000000",
        "classification": {
            "division": "STEM",
            "primary_domain": "general",
            "subdomains": [],
        },
        "source_artifacts": {
            "pdf": [{
                "path": str(pdf_path),
                "checksum": _sha256(pdf_bytes),
                "size": len(pdf_bytes),
            }],
        },
        "provenance": {
            "source_type": "textbook_to_course_pipeline",
            "import_pipeline_version": "1.0.0",
        },
        "features": {
            "source_provenance": True,
            "evidence_source_provenance": True,
        },
        # No concept_graph_sha256 + no concept_graph/ subdir at all.
        # Phase 7c ST 17: dart/imscc chunkset hashes are now required;
        # stub with synthetic 64-hex values so the new MISSING_*
        # checks don't pollute legacy-archive assertions below.
        "dart_chunks_sha256": "d" * 64,
        "imscc_chunks_sha256": "1" * 64,
    }
    manifest_path = course_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, course_dir


# ---------------------------------------------------------------------- #
# Test cases
# ---------------------------------------------------------------------- #


def test_concept_graph_sha256_regex_matches_canonical_hex():
    """Schema regex round-trip — must accept exactly 64 lowercase hex chars."""
    valid = "a" * 64
    assert _CONCEPT_GRAPH_SHA256_RE.match(valid)
    real_hash = hashlib.sha256(b"phase6 fixture").hexdigest()
    assert _CONCEPT_GRAPH_SHA256_RE.match(real_hash)
    assert not _CONCEPT_GRAPH_SHA256_RE.match("A" * 64)  # uppercase rejected
    assert not _CONCEPT_GRAPH_SHA256_RE.match("a" * 63)  # too short
    assert not _CONCEPT_GRAPH_SHA256_RE.match("a" * 65)  # too long
    assert not _CONCEPT_GRAPH_SHA256_RE.match("g" * 64)  # non-hex


def test_valid_concept_graph_hash_passes_no_warning(archive_with_graph):
    """Manifest with valid hash + matching graph emits no concept warnings."""
    manifest_path, course_dir, _graph_hash, _ = archive_with_graph
    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })
    cg_codes = {
        "MISSING_CONCEPT_GRAPH_SHA256",
        "INVALID_CONCEPT_GRAPH_SHA256",
        "CONCEPT_GRAPH_HASH_MISMATCH",
        "CONCEPT_GRAPH_READ_ERROR",
    }
    fired = [i.code for i in result.issues if i.code in cg_codes]
    assert not fired, (
        f"Valid concept_graph_sha256 must not fire any concept warnings; "
        f"got: {fired}"
    )
    # Concept-graph extension must not fire critical issues against a
    # well-formed concept_graph_sha256 + matching graph file. (Other
    # critical issues — e.g. from the new dart/imscc chunkset checks —
    # are out of scope for this concept-graph-focused test.)
    cg_codes = {
        "MISSING_CONCEPT_GRAPH_SHA256",
        "INVALID_CONCEPT_GRAPH_SHA256",
        "CONCEPT_GRAPH_HASH_MISMATCH",
        "CONCEPT_GRAPH_READ_ERROR",
    }
    cg_critical = [
        i for i in result.issues
        if i.severity == "critical" and i.code in cg_codes
    ]
    assert not cg_critical, (
        f"Concept-graph extension must not fire critical issues against "
        f"a valid hash + matching graph file; got: {[i.code for i in cg_critical]}"
    )


def test_legacy_archive_without_concept_graph_blocks_critical(archive_without_graph):
    """Legacy archive missing concept_graph_sha256 → critical fires (Phase 7c ST 17).

    Phase 6 left the gate at warning severity so legacy / DART-only archives
    didn't surface noise. Phase 7c ST 17 promoted the field to a required
    manifest key; legacy archives MUST now be backfilled (via
    ``LibV2/tools/libv2/scripts/backfill_dart_chunks.py``) to be valid.
    """
    manifest_path, course_dir = archive_without_graph
    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })
    missing = [
        i for i in result.issues
        if i.code == "MISSING_CONCEPT_GRAPH_SHA256"
    ]
    assert missing, (
        "Phase 7c ST 17: legacy archive without concept_graph_sha256 "
        "must fire MISSING_CONCEPT_GRAPH_SHA256 (now required)."
    )
    assert missing[0].severity == "critical", (
        "Phase 7c ST 17: MISSING_CONCEPT_GRAPH_SHA256 is critical-severity."
    )


def test_missing_hash_with_graph_present_blocks_critical(archive_with_graph):
    """Graph file exists but manifest lacks the hash → critical, blocks (Phase 7c ST 17)."""
    manifest_path, course_dir, _graph_hash, _graph_file = archive_with_graph
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("concept_graph_sha256", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })

    missing = [
        i for i in result.issues
        if i.code == "MISSING_CONCEPT_GRAPH_SHA256"
    ]
    assert missing, (
        "MISSING_CONCEPT_GRAPH_SHA256 must fire when graph exists but "
        "hash is absent."
    )
    assert missing[0].severity == "critical", (
        "Phase 7c ST 17 contract: MISSING_CONCEPT_GRAPH_SHA256 is critical-severity "
        "(promoted from Phase 6 warning)."
    )
    # Phase 7c contract: critical blocks.
    assert not result.passed, (
        "Phase 7c ST 17: missing concept_graph_sha256 MUST block (critical)."
    )


def test_invalid_hash_format_blocks_critical(archive_with_graph):
    """Malformed hash (wrong length / non-hex) → critical (Phase 7c ST 17)."""
    manifest_path, course_dir, _graph_hash, _ = archive_with_graph
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["concept_graph_sha256"] = "not-a-valid-sha256"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })

    invalid = [
        i for i in result.issues if i.code == "INVALID_CONCEPT_GRAPH_SHA256"
    ]
    assert invalid, (
        "INVALID_CONCEPT_GRAPH_SHA256 must fire on malformed hash."
    )
    assert invalid[0].severity == "critical", (
        "Phase 7c ST 17: INVALID_CONCEPT_GRAPH_SHA256 is critical."
    )
    # Phase 7c contract: critical blocks.
    assert not result.passed, (
        "Phase 7c ST 17: malformed concept_graph_sha256 MUST block."
    )


def test_hash_mismatch_blocks_critical(archive_with_graph):
    """Manifest hash diverges from on-disk graph bytes → critical (Phase 7c ST 17)."""
    manifest_path, course_dir, _graph_hash, graph_file = archive_with_graph

    # Tamper with the graph file AFTER manifest records the hash.
    original = graph_file.read_bytes()
    graph_file.write_bytes(original + b'\n{"tampered": true}\n')

    result = LibV2ManifestValidator().validate({
        "manifest_path": str(manifest_path),
        "course_dir": str(course_dir),
    })

    mismatch = [
        i for i in result.issues if i.code == "CONCEPT_GRAPH_HASH_MISMATCH"
    ]
    assert mismatch, (
        "CONCEPT_GRAPH_HASH_MISMATCH must fire when on-disk bytes diverge."
    )
    assert mismatch[0].severity == "critical", (
        "Phase 7c ST 17: CONCEPT_GRAPH_HASH_MISMATCH is critical."
    )
    # Phase 7c contract: critical blocks.
    assert not result.passed, (
        "Phase 7c ST 17: hash-mismatch MUST block."
    )


def test_archive_to_libv2_persists_concept_graph_sha256_to_manifest(tmp_path):
    """End-to-end: ``_archive_to_libv2`` routes the kwarg into manifest.

    Phase 6 ST 18 plumbed ``concept_graph_sha256`` from the
    ``concept_extraction`` phase output through the workflow runner's
    ``inputs_from`` chain into ``_archive_to_libv2``. This test
    invokes the helper directly and asserts the field lands in
    ``manifest.json``.
    """
    import asyncio
    import re
    from MCP.tools import pipeline_tools as pt

    # Redirect PROJECT_ROOT so the helper writes under tmp_path.
    fake_root = tmp_path / "root"
    fake_root.mkdir()
    (fake_root / "LibV2" / "courses").mkdir(parents=True)

    # Need to monkeypatch via setattr because the helper closure
    # captures PROJECT_ROOT at call time (not import time).
    orig_root = pt.PROJECT_ROOT
    pt.PROJECT_ROOT = fake_root
    try:
        registry = pt._build_tool_registry()
        archive = registry["archive_to_libv2"]

        # Stable canonical hash (64 lowercase hex).
        canonical_hash = hashlib.sha256(b"fixture concept graph").hexdigest()

        result = asyncio.run(archive(
            course_name="PHASE6_E2E",
            domain="general",
            division="STEM",
            concept_graph_sha256=canonical_hash,
        ))
        payload = json.loads(result)
        assert payload.get("success"), (
            f"archive_to_libv2 should succeed; got: {payload}"
        )

        manifest_path = Path(payload["manifest_path"])
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest.get("concept_graph_sha256") == canonical_hash, (
            f"manifest should carry the threaded hash; got "
            f"{manifest.get('concept_graph_sha256')!r}"
        )
        assert re.match(r"^[0-9a-f]{64}$", manifest["concept_graph_sha256"])
    finally:
        pt.PROJECT_ROOT = orig_root


def test_archive_to_libv2_persists_all_three_chunkset_hashes(tmp_path):
    """Phase 7c.5 SHIPPING BLOCKER end-to-end: ``_archive_to_libv2`` routes
    all three chunkset hashes (``dart_chunks_sha256`` +
    ``imscc_chunks_sha256`` + ``concept_graph_sha256``) into manifest.

    Phase 7c ST 17 promoted ``dart_chunks_sha256`` + ``imscc_chunks_sha256``
    to required manifest fields at the validator boundary. Phase 7c.5
    closes the producer-side gap: the workflow runner threads both hashes
    via ``inputs_from`` (``chunking`` → ``dart_chunks_sha256``,
    ``imscc_chunking`` → ``imscc_chunks_sha256``) into the same kwarg
    chain that ST 18 wired for ``concept_graph_sha256``.

    This test asserts the producer side: when all three kwargs are
    supplied, all three land in ``manifest.json`` with the supplied
    canonical 64-hex shape.
    """
    import asyncio
    import re
    from MCP.tools import pipeline_tools as pt

    fake_root = tmp_path / "root"
    fake_root.mkdir()
    (fake_root / "LibV2" / "courses").mkdir(parents=True)

    orig_root = pt.PROJECT_ROOT
    pt.PROJECT_ROOT = fake_root
    try:
        registry = pt._build_tool_registry()
        archive = registry["archive_to_libv2"]

        # Three distinct stable canonical hashes so the test catches a
        # cross-wiring bug (e.g. dart kwarg landing on imscc field).
        dart_hash = hashlib.sha256(b"fixture dart chunkset").hexdigest()
        imscc_hash = hashlib.sha256(b"fixture imscc chunkset").hexdigest()
        cg_hash = hashlib.sha256(b"fixture concept graph").hexdigest()
        assert dart_hash != imscc_hash != cg_hash

        result = asyncio.run(archive(
            course_name="PHASE7C5_E2E",
            domain="general",
            division="STEM",
            dart_chunks_sha256=dart_hash,
            imscc_chunks_sha256=imscc_hash,
            concept_graph_sha256=cg_hash,
        ))
        payload = json.loads(result)
        assert payload.get("success"), (
            f"archive_to_libv2 should succeed; got: {payload}"
        )

        manifest_path = Path(payload["manifest_path"])
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # All three hashes present on the canonical 64-hex shape.
        assert manifest.get("dart_chunks_sha256") == dart_hash, (
            f"manifest should carry the threaded dart hash; got "
            f"{manifest.get('dart_chunks_sha256')!r}"
        )
        assert manifest.get("imscc_chunks_sha256") == imscc_hash, (
            f"manifest should carry the threaded imscc hash; got "
            f"{manifest.get('imscc_chunks_sha256')!r}"
        )
        assert manifest.get("concept_graph_sha256") == cg_hash, (
            f"manifest should carry the threaded concept_graph hash; got "
            f"{manifest.get('concept_graph_sha256')!r}"
        )
        for field in (
            "dart_chunks_sha256", "imscc_chunks_sha256", "concept_graph_sha256",
        ):
            assert re.match(r"^[0-9a-f]{64}$", manifest[field]), (
                f"manifest[{field!r}] must satisfy the 64-hex regex"
            )
    finally:
        pt.PROJECT_ROOT = orig_root


def test_archive_to_libv2_omits_chunkset_hashes_when_kwargs_absent(tmp_path):
    """Backward compat: legacy callers that don't supply the new kwargs
    still produce a valid manifest (just without the new fields).

    The validator's ``MISSING_*`` critical fires downstream — that's the
    intended behavior for legacy / DART-only runs. The producer must NOT
    silently emit a placeholder hash that would mask the gap; it must
    omit the field so the validator's ``MISSING_*`` fires loudly.
    """
    import asyncio
    from MCP.tools import pipeline_tools as pt

    fake_root = tmp_path / "root"
    fake_root.mkdir()
    (fake_root / "LibV2" / "courses").mkdir(parents=True)

    orig_root = pt.PROJECT_ROOT
    pt.PROJECT_ROOT = fake_root
    try:
        registry = pt._build_tool_registry()
        archive = registry["archive_to_libv2"]

        result = asyncio.run(archive(
            course_name="PHASE7C5_LEGACY",
            domain="general",
            division="STEM",
        ))
        payload = json.loads(result)
        assert payload.get("success"), (
            f"archive_to_libv2 should succeed without new kwargs; got: {payload}"
        )

        manifest_path = Path(payload["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # New fields absent — caller didn't supply them, helper must
        # not invent values.
        assert "dart_chunks_sha256" not in manifest, (
            "Legacy caller without dart_chunks_sha256 kwarg must NOT see "
            "the field in manifest (would mask MISSING_* critical)."
        )
        assert "imscc_chunks_sha256" not in manifest, (
            "Legacy caller without imscc_chunks_sha256 kwarg must NOT see "
            "the field in manifest (would mask MISSING_* critical)."
        )
    finally:
        pt.PROJECT_ROOT = orig_root


def test_archive_to_libv2_rejects_malformed_chunkset_hashes(tmp_path):
    """Malformed hashes fall through to MISSING_* (not silently emitted).

    The helper validates against ``^[0-9a-f]{64}$`` and only writes the
    field when the kwarg matches. A malformed value (wrong length /
    uppercase / non-hex) is dropped on the producer side so the
    validator emits the same ``MISSING_*`` critical it would for a
    legacy archive — operators see the gap, not a downstream
    ``INVALID_*`` arising from corrupted producer-side data.
    """
    import asyncio
    from MCP.tools import pipeline_tools as pt

    fake_root = tmp_path / "root"
    fake_root.mkdir()
    (fake_root / "LibV2" / "courses").mkdir(parents=True)

    orig_root = pt.PROJECT_ROOT
    pt.PROJECT_ROOT = fake_root
    try:
        registry = pt._build_tool_registry()
        archive = registry["archive_to_libv2"]

        result = asyncio.run(archive(
            course_name="PHASE7C5_MALFORMED",
            domain="general",
            division="STEM",
            dart_chunks_sha256="not-a-valid-sha256",
            imscc_chunks_sha256="A" * 64,  # uppercase rejected by 64-hex regex
        ))
        payload = json.loads(result)
        assert payload.get("success"), payload

        manifest_path = Path(payload["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "dart_chunks_sha256" not in manifest
        assert "imscc_chunks_sha256" not in manifest
    finally:
        pt.PROJECT_ROOT = orig_root

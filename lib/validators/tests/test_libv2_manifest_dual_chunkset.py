"""Phase 7c Subtask 20 — end-to-end integration tests for the
DART + IMSCC dual-chunkset chain.

This is the **integration-scope** counterpart to the unit-level
coverage already landed in:

  - ``MCP/tests/test_pipeline_tools.py`` § ``TestRunDartChunkingEmitsChunksJsonl``
    + ``TestRunImsccChunkingEmitsChunksJsonl`` (Phase 7b ST 11 + Phase
    7c ST 16) — verify each chunker helper emits valid JSONL +
    sidecar manifest for a synthetic fixture.
  - ``lib/validators/tests/test_libv2_manifest_concept_graph.py``
    (Phase 6 ST 19, Phase 7c ST 17 promotion) — verify the
    ``concept_graph_sha256`` field's MISSING / INVALID / MISMATCH
    issue triplet.
  - ``lib/validators/tests/test_chunkset_manifest.py`` (Phase 7b
    ST 13) — verify ``ChunksetManifestValidator``'s schema +
    on-disk hash agreement checks.
  - ``LibV2/tests/test_backfill_dart_chunks.py`` (Phase 7c ST 18)
    — verify the operator backfill script for legacy archives.

What this test file adds beyond those is the **chain test**:
``_run_dart_chunking`` → ``_run_imscc_chunking`` → manifest construction
→ ``LibV2ManifestValidator``. The asserts close the loop on three
properties the unit tests can't reach individually:

  1. Both chunker helpers, run back-to-back against the same fake
     LibV2 root, land their respective ``chunks.jsonl`` files at
     paths that the LibV2 manifest validator's
     ``_check_dart_chunks_sha256`` + ``_check_imscc_chunks_sha256``
     find via ``course_dir / "<kind>_chunks" / "chunks.jsonl"``.
  2. A manually-constructed ``manifest.json`` that records the SHA-256
     fields from the chunker helpers' return envelopes (plus a stub
     concept_graph hash) passes the full ``LibV2ManifestValidator``
     gate at critical severity (i.e. zero critical issues against the
     three Phase 7c ST 17 hash-triangle checks).
  3. Tampering with any of the three on-disk artifacts after the
     manifest is sealed fires the corresponding ``*_HASH_MISMATCH``
     critical and flips ``passed`` to ``False`` — confirming the
     fail-closed posture for each leg of the triangle.

The test exercises the **post-archival** validator path; the
intermediate workflow plumbing (which would carry the chunk-set
hashes into ``_archive_to_libv2`` via ``inputs_from``) is currently
unwired (Phase 7d wiring follow-up). The test compensates by
constructing the manifest dict explicitly — once Phase 7d threads
the kwargs in, this test is the canonical regression for the
end-state contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest

# Project root for imports (pytest may not have project root on path).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib.validators.libv2_manifest import LibV2ManifestValidator  # noqa: E402
from MCP.tools import pipeline_tools  # noqa: E402


_DART_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title></head>
<body>
  <main>
    <section>
      <h1>{title}</h1>
      <p>This DART HTML file is a fixture for the Phase 7c integration
         smoke test. {body_pad}</p>
      <h2>Sub-section about pedagogy</h2>
      <p>Pedagogy describes the methods and practice of teaching.
         {tail_pad}</p>
    </section>
  </main>
</body>
</html>
"""


def _dart_html_payload(title: str) -> str:
    return _DART_HTML.format(
        title=title,
        body_pad=" ".join(["Chunk content padding sentence."] * 60),
        tail_pad=" ".join(["Additional padding to clear the chunker minimum-size threshold."] * 60),
    )


def _build_imscc_zip(zip_path: Path, html_files: list[tuple[str, str]]) -> None:
    """Build a minimal IMSCC zip at ``zip_path``."""
    import zipfile

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "imsmanifest.xml",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1">'
            '</manifest>\n',
        )
        for inner_path, html in html_files:
            zf.writestr(inner_path, html)


@pytest.fixture
def chunkset_chain_fixture(tmp_path, monkeypatch):
    """Run both chunker helpers against synthetic inputs, return the
    fake LibV2 root + the JSON envelopes from each helper.

    Lays down on disk:
      * ``<fake_root>/LibV2/courses/<slug>/dart_chunks/chunks.jsonl + manifest.json``
      * ``<fake_root>/LibV2/courses/<slug>/imscc_chunks/chunks.jsonl + manifest.json``

    Both helpers use ``pipeline_tools._PROJECT_ROOT`` to resolve the
    LibV2 course directory (per Phase 7b ST 11 + Phase 7c ST 16); we
    monkeypatch that module-level constant to the temp tree so the
    test stays hermetic.
    """
    fake_root = tmp_path / "root"
    fake_root.mkdir()

    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
    monkeypatch.setattr(
        pipeline_tools,
        "COURSEFORGE_INPUTS",
        fake_root / "Courseforge" / "inputs" / "textbooks",
    )
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)

    # --- DART staging dir + two HTML files --------------------------
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "chapter_01.html").write_text(
        _dart_html_payload("Chapter One"), encoding="utf-8"
    )
    (staging / "chapter_02.html").write_text(
        _dart_html_payload("Chapter Two"), encoding="utf-8"
    )

    # --- Packaged IMSCC -------------------------------------------------
    imscc_path = tmp_path / "course.imscc"
    _build_imscc_zip(
        imscc_path,
        [
            ("html/page_01.html", _dart_html_payload("Page One")),
            ("html/page_02.html", _dart_html_payload("Page Two")),
        ],
    )

    course_name = "PHASE7C_E2E"
    course_slug = "phase7c-e2e"

    registry = pipeline_tools._build_tool_registry()

    dart_payload = json.loads(asyncio.run(
        registry["run_dart_chunking"](
            course_name=course_name,
            staging_dir=str(staging),
        )
    ))
    imscc_payload = json.loads(asyncio.run(
        registry["run_imscc_chunking"](
            course_name=course_name,
            imscc_path=str(imscc_path),
        )
    ))

    return {
        "fake_root": fake_root,
        "course_name": course_name,
        "course_slug": course_slug,
        "course_dir": fake_root / "LibV2" / "courses" / course_slug,
        "dart_payload": dart_payload,
        "imscc_payload": imscc_payload,
    }


def _seal_manifest(
    course_dir: Path,
    *,
    dart_chunks_sha256: str,
    imscc_chunks_sha256: str,
    concept_graph_sha256: str,
    pdf_bytes: bytes = b"%PDF-1.4 phase7c integration fixture",
) -> Path:
    """Write a minimal LibV2 ``manifest.json`` carrying all three
    Phase 7c required hashes. Mirrors the shape ``_archive_to_libv2``
    emits today (modulo the new chunkset hashes — which Phase 7d
    threads in via ``inputs_from``; this helper anticipates that wiring).
    """
    # Source PDF artifact so source_artifacts integrity check has work.
    pdf_path = course_dir / "source" / "pdf" / "fixture.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)

    # Concept graph file matching the supplied hash.
    cg_dir = course_dir / "concept_graph"
    cg_dir.mkdir(parents=True, exist_ok=True)
    # We don't fabricate bytes that hash to ``concept_graph_sha256`` —
    # instead, we recompute the hash from concrete bytes and let the
    # caller pass the result back. The fixture below uses this pattern.
    # When this helper is called WITHOUT pre-existing graph bytes, the
    # bytes are constructed deterministically here so the hash is
    # reproducible.

    manifest: Dict[str, Any] = {
        "libv2_version": "1.2.0",
        "chunker_version": "1.0.0+phase7c-test",
        "slug": course_dir.name,
        "import_timestamp": datetime.utcnow().isoformat(),
        "classification": {
            "division": "STEM",
            "primary_domain": "general",
            "subdomains": [],
        },
        "source_artifacts": {
            "pdf": [{
                "path": str(pdf_path),
                "checksum": hashlib.sha256(pdf_bytes).hexdigest(),
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
        "dart_chunks_sha256": dart_chunks_sha256,
        "imscc_chunks_sha256": imscc_chunks_sha256,
        "concept_graph_sha256": concept_graph_sha256,
    }
    manifest_path = course_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


@pytest.fixture
def sealed_archive(chunkset_chain_fixture):
    """Extend ``chunkset_chain_fixture`` with a sealed ``manifest.json``
    that records all three Phase 7c required hashes.

    The concept-graph leg of the triangle is fabricated here (the
    chunker helpers don't emit it; the Phase 6 concept_extraction
    phase does). We write deterministic bytes + record the matching
    SHA so the round-trip check passes.
    """
    fx = chunkset_chain_fixture
    course_dir = fx["course_dir"]

    # Concept graph fixture bytes + hash.
    cg_payload = {
        "kind": "pedagogy",
        "course_id": fx["course_name"],
        "nodes": [],
        "edges": [],
    }
    cg_bytes = json.dumps(cg_payload, indent=2).encode("utf-8")
    cg_hash = hashlib.sha256(cg_bytes).hexdigest()
    cg_dir = course_dir / "concept_graph"
    cg_dir.mkdir(parents=True, exist_ok=True)
    (cg_dir / "concept_graph_semantic.json").write_bytes(cg_bytes)

    manifest_path = _seal_manifest(
        course_dir,
        dart_chunks_sha256=fx["dart_payload"]["dart_chunks_sha256"],
        imscc_chunks_sha256=fx["imscc_payload"]["imscc_chunks_sha256"],
        concept_graph_sha256=cg_hash,
    )
    return {
        **fx,
        "manifest_path": manifest_path,
        "concept_graph_sha256": cg_hash,
    }


# ---------------------------------------------------------------------- #
# Test cases
# ---------------------------------------------------------------------- #


class TestEndToEndChainEmits:
    """Phase 7b/c chain emits all three artifacts to the LibV2 layout."""

    def test_dart_chunks_artifacts_landed(self, chunkset_chain_fixture):
        fx = chunkset_chain_fixture
        course_dir = fx["course_dir"]
        chunks_path = course_dir / "dart_chunks" / "chunks.jsonl"
        manifest_path = course_dir / "dart_chunks" / "manifest.json"
        assert chunks_path.is_file(), (
            f"Phase 7b chunkset missing at {chunks_path}"
        )
        assert manifest_path.is_file(), (
            f"Phase 7b sidecar missing at {manifest_path}"
        )
        # Round-trip: helper-returned hash matches on-disk bytes.
        on_disk = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        assert on_disk == fx["dart_payload"]["dart_chunks_sha256"]

    def test_imscc_chunks_artifacts_landed(self, chunkset_chain_fixture):
        fx = chunkset_chain_fixture
        course_dir = fx["course_dir"]
        chunks_path = course_dir / "imscc_chunks" / "chunks.jsonl"
        manifest_path = course_dir / "imscc_chunks" / "manifest.json"
        assert chunks_path.is_file(), (
            f"Phase 7c chunkset missing at {chunks_path}"
        )
        assert manifest_path.is_file(), (
            f"Phase 7c sidecar missing at {manifest_path}"
        )
        on_disk = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        assert on_disk == fx["imscc_payload"]["imscc_chunks_sha256"]

    def test_chunkset_kinds_distinguish(self, chunkset_chain_fixture):
        """Each chunkset's manifest carries the canonical kind discriminator."""
        fx = chunkset_chain_fixture
        course_dir = fx["course_dir"]
        dart_manifest = json.loads(
            (course_dir / "dart_chunks" / "manifest.json").read_text(encoding="utf-8")
        )
        imscc_manifest = json.loads(
            (course_dir / "imscc_chunks" / "manifest.json").read_text(encoding="utf-8")
        )
        assert dart_manifest["chunkset_kind"] == "dart"
        assert "source_dart_html_sha256" in dart_manifest
        assert imscc_manifest["chunkset_kind"] == "imscc"
        assert "source_imscc_sha256" in imscc_manifest


class TestSealedManifestPassesValidator:
    """The sealed three-hash manifest passes ``LibV2ManifestValidator``."""

    def test_three_required_fields_present(self, sealed_archive):
        manifest = json.loads(
            sealed_archive["manifest_path"].read_text(encoding="utf-8")
        )
        for field in (
            "dart_chunks_sha256",
            "imscc_chunks_sha256",
            "concept_graph_sha256",
        ):
            assert field in manifest, f"missing required field: {field}"
            assert isinstance(manifest[field], str)
            assert len(manifest[field]) == 64

    def test_validator_accepts_clean_archive(self, sealed_archive):
        """No critical issues from any of the three chunkset / graph
        gates (the only critical issues would be from out-of-scope
        non-Phase-7c surfaces, which is acceptable for this test)."""
        result = LibV2ManifestValidator().validate({
            "manifest_path": str(sealed_archive["manifest_path"]),
            "course_dir": str(sealed_archive["course_dir"]),
        })
        triangle_codes = {
            "MISSING_DART_CHUNKS_SHA256",
            "INVALID_DART_CHUNKS_SHA256",
            "DART_CHUNKS_HASH_MISMATCH",
            "MISSING_IMSCC_CHUNKS_SHA256",
            "INVALID_IMSCC_CHUNKS_SHA256",
            "IMSCC_CHUNKS_HASH_MISMATCH",
            "MISSING_CONCEPT_GRAPH_SHA256",
            "INVALID_CONCEPT_GRAPH_SHA256",
            "CONCEPT_GRAPH_HASH_MISMATCH",
        }
        triangle_critical = [
            i for i in result.issues
            if i.severity == "critical" and i.code in triangle_codes
        ]
        assert not triangle_critical, (
            f"Phase 7c ST 17 chunkset triangle must not fire critical "
            f"issues against a clean archive; got: "
            f"{[i.code for i in triangle_critical]}"
        )


class TestTamperingFiresHashMismatch:
    """Tampering with any leg of the triangle fires the matching critical."""

    def test_tampered_dart_chunks_jsonl_fires_mismatch(self, sealed_archive):
        chunks_path = (
            sealed_archive["course_dir"] / "dart_chunks" / "chunks.jsonl"
        )
        original = chunks_path.read_bytes()
        chunks_path.write_bytes(original + b'{"tampered": true}\n')

        result = LibV2ManifestValidator().validate({
            "manifest_path": str(sealed_archive["manifest_path"]),
            "course_dir": str(sealed_archive["course_dir"]),
        })
        mismatches = [
            i for i in result.issues
            if i.code == "DART_CHUNKS_HASH_MISMATCH"
        ]
        assert mismatches, (
            "tampering with dart_chunks/chunks.jsonl must fire "
            "DART_CHUNKS_HASH_MISMATCH (Phase 7c ST 17)"
        )
        assert mismatches[0].severity == "critical"
        assert not result.passed, (
            "Phase 7c ST 17: dart-chunks tampering MUST block (critical)."
        )

    def test_tampered_imscc_chunks_jsonl_fires_mismatch(self, sealed_archive):
        chunks_path = (
            sealed_archive["course_dir"] / "imscc_chunks" / "chunks.jsonl"
        )
        original = chunks_path.read_bytes()
        chunks_path.write_bytes(original + b'{"tampered": true}\n')

        result = LibV2ManifestValidator().validate({
            "manifest_path": str(sealed_archive["manifest_path"]),
            "course_dir": str(sealed_archive["course_dir"]),
        })
        mismatches = [
            i for i in result.issues
            if i.code == "IMSCC_CHUNKS_HASH_MISMATCH"
        ]
        assert mismatches, (
            "tampering with imscc_chunks/chunks.jsonl must fire "
            "IMSCC_CHUNKS_HASH_MISMATCH (Phase 7c ST 17)"
        )
        assert mismatches[0].severity == "critical"
        assert not result.passed, (
            "Phase 7c ST 17: imscc-chunks tampering MUST block (critical)."
        )

    def test_tampered_concept_graph_fires_mismatch(self, sealed_archive):
        cg_path = (
            sealed_archive["course_dir"]
            / "concept_graph"
            / "concept_graph_semantic.json"
        )
        original = cg_path.read_bytes()
        cg_path.write_bytes(original + b'\n{"tampered": true}\n')

        result = LibV2ManifestValidator().validate({
            "manifest_path": str(sealed_archive["manifest_path"]),
            "course_dir": str(sealed_archive["course_dir"]),
        })
        mismatches = [
            i for i in result.issues
            if i.code == "CONCEPT_GRAPH_HASH_MISMATCH"
        ]
        assert mismatches, (
            "tampering with concept_graph_semantic.json must fire "
            "CONCEPT_GRAPH_HASH_MISMATCH (Phase 7c ST 17 promotion)"
        )
        assert mismatches[0].severity == "critical"
        assert not result.passed, (
            "Phase 7c ST 17: concept-graph tampering MUST block (critical)."
        )


class TestMissingChunksetFieldBlocks:
    """Stripping a required hash from the manifest fires MISSING_*."""

    def test_missing_dart_chunks_sha256_fires_critical(self, sealed_archive):
        manifest = json.loads(
            sealed_archive["manifest_path"].read_text(encoding="utf-8")
        )
        manifest.pop("dart_chunks_sha256")
        sealed_archive["manifest_path"].write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        result = LibV2ManifestValidator().validate({
            "manifest_path": str(sealed_archive["manifest_path"]),
            "course_dir": str(sealed_archive["course_dir"]),
        })
        missing = [
            i for i in result.issues
            if i.code == "MISSING_DART_CHUNKS_SHA256"
        ]
        assert missing, (
            "MISSING_DART_CHUNKS_SHA256 must fire when field absent."
        )
        assert missing[0].severity == "critical"
        assert not result.passed

    def test_missing_imscc_chunks_sha256_fires_critical(self, sealed_archive):
        manifest = json.loads(
            sealed_archive["manifest_path"].read_text(encoding="utf-8")
        )
        manifest.pop("imscc_chunks_sha256")
        sealed_archive["manifest_path"].write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        result = LibV2ManifestValidator().validate({
            "manifest_path": str(sealed_archive["manifest_path"]),
            "course_dir": str(sealed_archive["course_dir"]),
        })
        missing = [
            i for i in result.issues
            if i.code == "MISSING_IMSCC_CHUNKS_SHA256"
        ]
        assert missing, (
            "MISSING_IMSCC_CHUNKS_SHA256 must fire when field absent."
        )
        assert missing[0].severity == "critical"
        assert not result.passed

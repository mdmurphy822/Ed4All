"""End-to-end textbook-to-course pipeline integration test.

This test **gates the parallel-worker completion** of the pipeline-execution-
fixes milestone. It runs ``ed4all run textbook-to-course`` against the
committed fixture PDF and asserts all three output contracts:

  * Worker α — Courseforge content-generator — real 5-page weekly modules
    with ``data-cf-*`` + JSON-LD, not the ``DIGPED 101`` hardcoded template.
  * Worker β — Trainforge assessment — ``chunks.jsonl`` +
    ``concept_graph_semantic.json`` + ``misconceptions.json`` land on disk
    with the right shapes.
  * Worker γ — LibV2 archival — ``corpus/``, ``graph/`` populated;
    ``manifest.features.source_provenance`` key present.

**EXPECTED TO FAIL TODAY.** Until all three workers land, at least one of
the assertions below will fail. That's the point — this test is the
completion signal. The default pytest run skips it via the ``slow`` marker.

Run it explicitly with:

    pytest -m slow tests/integration/test_pipeline_end_to_end.py

The test uses the opt-in Wave 7–11 strict/stable-ID flags so the fixture
output exercises the same code paths that production runs will hit once
the three workers ship.

Contract source of truth: ``plans/pipeline-execution-fixes/contracts.md``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_PDF = PROJECT_ROOT / "tests" / "fixtures" / "pipeline" / "fixture_corpus.pdf"

# Matches Courseforge's slug convention:
# course_name.lower().replace("_", "-").replace(" ", "-").
COURSE_NAME = "TESTPIPE_101"
COURSE_SLUG = "testpipe-101"

# Opt-in Wave 7–11 flags (strict shapes, stable IDs, source provenance).
STRICT_ENV_FLAGS = {
    "TRAINFORGE_CONTENT_HASH_IDS": "true",
    "TRAINFORGE_SCOPE_CONCEPT_IDS": "true",
    "TRAINFORGE_PRESERVE_LO_CASE": "true",
    "TRAINFORGE_VALIDATE_CHUNKS": "true",
    "TRAINFORGE_ENFORCE_CONTENT_TYPE": "true",
    "TRAINFORGE_STRICT_EVIDENCE": "true",
    "TRAINFORGE_SOURCE_PROVENANCE": "true",
    "DECISION_VALIDATION_STRICT": "true",
}


# ---------------------------------------------------------------------- #
# Cleanup helpers
# ---------------------------------------------------------------------- #


ARTIFACT_PATHS_TO_CLEAN: tuple[Path, ...] = (
    PROJECT_ROOT / "state" / "runs",
    PROJECT_ROOT / "Courseforge" / "exports",
    PROJECT_ROOT / "LibV2" / "courses" / COURSE_SLUG,
    PROJECT_ROOT / "training-captures" / "trainforge" / COURSE_NAME,
    PROJECT_ROOT / "training-captures" / "courseforge" / COURSE_NAME,
    PROJECT_ROOT / "training-captures" / "dart" / COURSE_NAME,
)


def _snapshot_existing() -> dict[Path, set[str]]:
    """Record existing entries at each cleanup target so we only remove
    artifacts *this run* produced. Keeps shared dirs (``state/runs/``,
    ``Courseforge/exports/``) intact for any other workflows in flight."""
    snapshot: dict[Path, set[str]] = {}
    for target in ARTIFACT_PATHS_TO_CLEAN:
        if target.exists() and target.is_dir():
            snapshot[target] = {p.name for p in target.iterdir()}
        else:
            snapshot[target] = set()
    return snapshot


def _cleanup_new(snapshot: dict[Path, set[str]]) -> None:
    """Remove entries added since ``snapshot`` was taken. Safe to call even
    if the run crashed and left partial output."""
    for target, existing in snapshot.items():
        if not target.exists():
            continue
        if target.name == COURSE_SLUG:
            # LibV2 course dir — remove outright; entire subtree belongs to this run.
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            continue
        if target.name == COURSE_NAME:
            # training-captures/{tool}/{course} — same story.
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            continue
        if not target.is_dir():
            continue
        for entry in target.iterdir():
            if entry.name not in existing:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    try:
                        entry.unlink()
                    except OSError:
                        pass


# ---------------------------------------------------------------------- #
# Pipeline runner
# ---------------------------------------------------------------------- #


def _run_ed4all_cli() -> subprocess.CompletedProcess:
    """Invoke ``ed4all run textbook-to-course`` via subprocess."""
    env = os.environ.copy()
    env.update(STRICT_ENV_FLAGS)
    # Force local mode so no ANTHROPIC_API_KEY is needed.
    env["LLM_MODE"] = "local"

    cmd = [
        sys.executable,
        "-m",
        "cli.main",
        "run",
        "textbook-to-course",
        "--corpus",
        str(FIXTURE_PDF),
        "--course-name",
        COURSE_NAME,
        "--weeks",
        "2",
        "--assessment-count",
        "6",
        "--mode",
        "local",
        "--json",
    ]
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


# ---------------------------------------------------------------------- #
# Per-worker assertions
# ---------------------------------------------------------------------- #


def _find_export_dir() -> Path:
    exports = PROJECT_ROOT / "Courseforge" / "exports"
    assert exports.exists(), "Courseforge/exports/ missing after pipeline run"
    # Find the most recently-modified project dir that mentions TESTPIPE.
    candidates = [
        p for p in exports.iterdir()
        if p.is_dir() and COURSE_NAME.lower() in p.name.lower()
    ]
    assert candidates, (
        f"No Courseforge/exports/ dir found for {COURSE_NAME}; "
        f"available: {[p.name for p in exports.iterdir()]}"
    )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _assert_worker_alpha(export_dir: Path) -> None:
    """Courseforge content-generator produced the 5-page weekly modules
    with full data-cf-* + JSON-LD metadata."""
    week_01_dir = export_dir / "03_content_development" / "week_01"
    assert week_01_dir.exists(), (
        f"Worker α: week_01 content dir missing at {week_01_dir}. "
        "Expected 5 HTML pages per week (overview/content/application/"
        "self_check/summary)."
    )

    html_files = sorted(week_01_dir.glob("*.html"))
    assert len(html_files) >= 5, (
        f"Worker α: week_01 has {len(html_files)} HTML pages, "
        f"expected >= 5 (overview/content/application/self_check/summary). "
        f"Files: {[f.name for f in html_files]}"
    )

    # First page must carry the full Courseforge metadata surface.
    first_page = html_files[0]
    html = first_page.read_text(encoding="utf-8")

    assert 'data-cf-role="template-chrome"' in html, (
        f"Worker α: {first_page.name} missing data-cf-role=template-chrome "
        "on skip-link/header/footer."
    )

    assert 'application/ld+json' in html, (
        f"Worker α: {first_page.name} missing <script type=application/ld+json> "
        "block. Required per courseforge_jsonld_v1.schema.json."
    )

    assert 'data-cf-objective-id=' in html, (
        f"Worker α: {first_page.name} has no data-cf-objective-id attributes. "
        "page_objectives gate requires >= 1 per page."
    )

    # Not the old hardcoded template. The stub produced "DIGPED 101" at
    # pipeline_tools.py:1396 — Worker α must replace that.
    assert 'DIGPED 101' not in html, (
        f"Worker α: {first_page.name} still contains the 'DIGPED 101' "
        "hardcoded template text — stub output was not replaced."
    )


def _assert_worker_beta(export_dir: Path) -> None:
    """Trainforge produced chunks.jsonl + concept_graph_semantic.json +
    misconceptions.json at the expected workspace location."""
    # The exact workspace path is Worker β's choice (contracts.md allows
    # either ``state/runs/{run_id}/trainforge/`` or ``{export_dir}/trainforge/``).
    # Search both plus LibV2 archive location.
    candidates = [
        export_dir / "trainforge",
        PROJECT_ROOT / "LibV2" / "courses" / COURSE_SLUG / "corpus",
    ]
    for run_dir in (PROJECT_ROOT / "state" / "runs").glob("*/trainforge"):
        candidates.append(run_dir)

    chunks_file: Path | None = None
    for candidate in candidates:
        candidate_chunks = candidate / "chunks.jsonl"
        if candidate_chunks.exists():
            chunks_file = candidate_chunks
            break

    assert chunks_file is not None, (
        f"Worker β: chunks.jsonl not found in any expected location. "
        f"Searched: {[str(c) for c in candidates]}"
    )

    lines = [
        line for line in chunks_file.read_text().splitlines() if line.strip()
    ]
    assert len(lines) >= 5, (
        f"Worker β: chunks.jsonl has {len(lines)} chunks, expected >= 5."
    )

    # Each chunk must validate under chunk_v4.schema.json strict.
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    schema_path = PROJECT_ROOT / "schemas" / "knowledge" / "chunk_v4.schema.json"
    srcref_path = (
        PROJECT_ROOT / "schemas" / "knowledge" / "source_reference.schema.json"
    )
    chunk_schema = json.loads(schema_path.read_text())
    srcref_schema = json.loads(srcref_path.read_text())
    resources = [
        (chunk_schema["$id"], Resource.from_contents(chunk_schema)),
        (srcref_schema["$id"], Resource.from_contents(srcref_schema)),
    ]
    tax_dir = PROJECT_ROOT / "schemas" / "taxonomies"
    for name in [
        "bloom_verbs.json", "content_type.json", "cognitive_domain.json",
    ]:
        tax = json.loads((tax_dir / name).read_text())
        resources.append((tax["$id"], Resource.from_contents(tax)))
    registry = Registry().with_resources(resources)
    validator = Draft202012Validator(chunk_schema, registry=registry)

    for i, line in enumerate(lines):
        chunk = json.loads(line)
        errors = sorted(validator.iter_errors(chunk), key=lambda e: list(e.path))
        assert not errors, (
            f"Worker β: chunk {i} ({chunk.get('id')}) fails chunk_v4.schema:\n"
            + "\n".join(f"    - {e.message}" for e in errors[:5])
        )

    # Concept graph landed and spans >= 2 edge types.
    graph_file = chunks_file.parent.parent / "graph" / "concept_graph_semantic.json"
    if not graph_file.exists():
        graph_file = chunks_file.parent / "concept_graph_semantic.json"
    assert graph_file.exists(), (
        f"Worker β: concept_graph_semantic.json not found near {chunks_file}."
    )
    graph = json.loads(graph_file.read_text())
    assert len(graph.get("edges", [])) >= 3, (
        f"Worker β: graph has {len(graph.get('edges', []))} edges, expected >= 3."
    )
    edge_types = {edge["type"] for edge in graph["edges"]}
    assert len(edge_types) >= 2, (
        f"Worker β: graph spans {len(edge_types)} edge types ({edge_types}), "
        "expected >= 2 distinct types."
    )

    # Misconceptions with proper ID shape.
    mc_file = graph_file.parent / "misconceptions.json"
    if not mc_file.exists():
        mc_file = chunks_file.parent / "misconceptions.json"
    assert mc_file.exists(), (
        f"Worker β: misconceptions.json not found near {graph_file}."
    )
    mc_doc = json.loads(mc_file.read_text())
    misconceptions = mc_doc.get("misconceptions") or mc_doc
    assert isinstance(misconceptions, list) and len(misconceptions) >= 1, (
        f"Worker β: misconceptions.json has no entries."
    )
    mc_id_re = re.compile(r"^mc_[0-9a-f]{16}$")
    has_valid_id = any(
        isinstance(m.get("id"), str) and mc_id_re.match(m["id"])
        for m in misconceptions
    )
    assert has_valid_id, (
        f"Worker β: no misconception has an id matching ^mc_[0-9a-f]{{16}}$. "
        f"IDs seen: {[m.get('id') for m in misconceptions]}"
    )


def _assert_worker_gamma() -> None:
    """LibV2 archival populated corpus/ + graph/ and wrote a proper manifest."""
    course_dir = PROJECT_ROOT / "LibV2" / "courses" / COURSE_SLUG
    assert course_dir.exists(), (
        f"Worker γ: LibV2 course dir missing at {course_dir}."
    )

    # Corpus must have chunks.jsonl (copied byte-for-byte from Trainforge output).
    corpus_chunks = course_dir / "corpus" / "chunks.jsonl"
    assert corpus_chunks.exists(), (
        f"Worker γ: {corpus_chunks} missing. archival should copy "
        "trainforge/chunks.jsonl into corpus/."
    )
    assert corpus_chunks.stat().st_size > 0, (
        f"Worker γ: {corpus_chunks} is empty."
    )

    # Graph dir populated.
    graph_dir = course_dir / "graph"
    graph_files = list(graph_dir.glob("*.json"))
    assert graph_files, (
        f"Worker γ: {graph_dir} has no JSON files. Expected at least "
        "concept_graph_semantic.json."
    )

    # Manifest present and carries features.source_provenance key (value
    # may be False if the stub source-router produced an empty map — the
    # key just has to exist).
    manifest_path = course_dir / "manifest.json"
    assert manifest_path.exists(), f"Worker γ: {manifest_path} missing."
    manifest = json.loads(manifest_path.read_text())
    assert "features" in manifest, (
        f"Worker γ: manifest.json has no 'features' block: "
        f"keys={list(manifest.keys())}"
    )
    assert "source_provenance" in manifest["features"], (
        f"Worker γ: manifest.features has no 'source_provenance' key: "
        f"features={manifest['features']}"
    )

    # No empty archival dirs. (``pedagogy/`` may be empty — that's
    # expected; contracts.md calls it out as a gap. Tolerate it.)
    for subdir_name in ("corpus", "graph"):
        subdir = course_dir / subdir_name
        assert subdir.exists() and any(subdir.iterdir()), (
            f"Worker γ: {subdir} exists but is empty."
        )


# ---------------------------------------------------------------------- #
# The test
# ---------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.integration
def test_textbook_to_course_end_to_end():
    """Run the full textbook-to-course pipeline + assert the three output
    contracts from plans/pipeline-execution-fixes/contracts.md.

    Expected to fail until workers α, β, γ all land.
    """
    assert FIXTURE_PDF.exists(), (
        f"Fixture PDF missing at {FIXTURE_PDF}. Run "
        "tests/fixtures/pipeline/build_fixture_pdf.py to regenerate."
    )

    snapshot = _snapshot_existing()
    try:
        result = _run_ed4all_cli()

        # Capture stdout/stderr for the failure message — it's the primary
        # diagnostic when the CLI itself crashed rather than when a
        # contract assertion failed.
        assert result.returncode == 0, (
            f"ed4all run failed with exit code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        export_dir = _find_export_dir()

        # Each worker's contract is asserted independently so failure
        # messages name the responsible worker directly.
        _assert_worker_alpha(export_dir)
        _assert_worker_beta(export_dir)
        _assert_worker_gamma()
    finally:
        _cleanup_new(snapshot)

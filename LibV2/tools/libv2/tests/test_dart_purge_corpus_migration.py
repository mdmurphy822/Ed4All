"""Tests for the DART->semantik corpus-body migration (naming purge Stage S2).

Exercises :func:`LibV2.tools.libv2.migrate.migrate_course_corpus` on a tiny
SYNTHETIC course fixture (no live slug, no fixture corpus): a course carrying
``dart:`` sourceIds + ``data-dart-*`` HTML attrs + the dart-named hashes, run in
BOTH dry-run (asserts nothing changed on disk) and apply (asserts sourceIds /
attrs / hashes / chunkset_kind all flipped, the dir renamed, and that the
dual-read LibV2 reader still resolves the migrated chunkset).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from LibV2.tools.libv2.migrate import (
    DART_PURGE_REGISTRY,
    build_dart_purge_registry,
    migrate_course_corpus,
    plan_dart_purge,
)
from lib.libv2_storage import resolve_imscc_chunks_path


# --------------------------------------------------------------------------- #
# Synthetic fixture — a dart-named course built at fixture-load time.         #
# --------------------------------------------------------------------------- #

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chunk_line(idx: int, slug: str) -> str:
    """One chunk_v4-ish JSONL line carrying a dart: sourceId + data-dart- html."""
    obj = {
        "id": f"{slug}#chunk_{idx:02d}",
        "chunk_type": "explanation",
        "text": f"Body text {idx} mentioning dartboards and darting about.",
        "html": (
            f'<section data-dart-block-id="b{idx:02d}" '
            f'data-dart-block-role="paragraph" '
            f'data-dart-source="dart_converter">para {idx}</section>'
        ),
        "source": {
            "course_id": slug.upper(),
            "source_references": [
                {
                    "sourceId": f"dart:{slug}#section-{idx}",
                    "role": "primary",
                    "extractor": "synthesized",
                }
            ],
        },
    }
    return json.dumps(obj, ensure_ascii=False)


def _make_dart_course(root: Path, slug: str = "synthetic-dart-course") -> Path:
    """Build a synthetic dart-named course under ``root/courses/<slug>``."""
    cdir = root / "courses" / slug
    dart = cdir / "dart_chunks"
    dart.mkdir(parents=True)

    lines = [_chunk_line(1, slug), _chunk_line(2, slug)]
    chunks_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    (dart / "chunks.jsonl").write_bytes(chunks_bytes)
    chunks_sha = _sha(chunks_bytes)

    # Sidecar chunkset manifest (dart-named).
    (dart / "manifest.json").write_text(
        json.dumps(
            {
                "chunks_sha256": chunks_sha,
                "chunker_version": "v4",
                "chunkset_kind": "dart",
                "source_dart_html_sha256": "d" * 64,
                "chunks_count": len(lines),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Archived HTML carrying data-dart-* attrs.
    html_dir = cdir / "source" / "html"
    html_dir.mkdir(parents=True)
    (html_dir / "page01_accessible.html").write_text(
        '<html><body><main data-dart-block-id="abc123" '
        'data-dart-source="dart_converter"><p data-dart-pages="2">hi</p>'
        "</main></body></html>",
        encoding="utf-8",
    )

    # Top-level course manifest with dart_chunks_sha256.
    (cdir / "manifest.json").write_text(
        json.dumps(
            {
                "libv2_version": "1.2.0",
                "slug": slug,
                "classification": {"division": "STEM", "primary_domain": "physics"},
                "content_profile": {"total_chunks": len(lines)},
                "dart_chunks_sha256": chunks_sha,
                "imscc_chunks_sha256": "i" * 64,
                "concept_graph_sha256": "c" * 64,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "catalog").mkdir(parents=True, exist_ok=True)
    return cdir


def _snapshot(cdir: Path) -> dict:
    """Byte-snapshot every file under a course dir (for no-write assertions)."""
    return {
        str(p.relative_to(cdir)): p.read_bytes()
        for p in sorted(cdir.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------- #
# Dry-run: computes the plan, writes nothing.                                 #
# --------------------------------------------------------------------------- #

def test_dry_run_writes_nothing_but_reports_the_plan(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_dart_course(root)
    before = _snapshot(cdir)

    report = migrate_course_corpus(cdir, apply=False)

    # Nothing on disk changed.
    assert _snapshot(cdir) == before
    assert not report.applied
    assert (cdir / "dart_chunks").is_dir()
    assert not (cdir / "semantik_chunks").exists()

    # The plan is accurate.
    assert report.applicable
    assert not report.already_migrated
    assert report.chunks_lines_total == 2
    assert report.chunks_lines_changed == 2
    assert report.sourceids_changed == 2           # one dart: per line
    assert report.chunk_html_attrs_changed == 6    # 3 data-dart- attrs x 2 lines
    assert report.html_files_total == 1
    assert report.html_files_changed == 1
    assert report.html_attrs_changed == 3          # 3 data-dart- attrs in the html
    assert report.sidecar_present
    assert report.sidecar_kind_flipped
    assert report.sidecar_source_key_renamed
    assert report.top_manifest_present
    assert report.top_manifest_key_renamed
    assert report.dir_renamed is True
    assert report.new_chunks_sha256 != report.old_chunks_sha256
    assert report.would_change_anything


def test_dry_run_no_dart_naming_is_advisory(tmp_path):
    """A chunkset with no dart: / data-dart- tokens reports nothing to rewrite."""
    root = tmp_path / "libv2"
    cdir = root / "courses" / "clean-course"
    dart = cdir / "dart_chunks"
    dart.mkdir(parents=True)
    (dart / "chunks.jsonl").write_text('{"id": "x", "text": "no tokens"}\n', encoding="utf-8")
    (dart / "manifest.json").write_text(
        json.dumps({"chunkset_kind": "dart", "chunks_sha256": "0" * 64}),
        encoding="utf-8",
    )

    report = migrate_course_corpus(cdir, apply=False)
    assert report.applicable
    assert report.sourceids_changed == 0
    assert report.chunk_html_attrs_changed == 0
    assert any("nothing to rewrite" in a for a in report.advisories)


def test_non_dart_course_not_applicable(tmp_path):
    root = tmp_path / "libv2"
    cdir = root / "courses" / "imscc-only"
    (cdir / "imscc_chunks").mkdir(parents=True)
    (cdir / "imscc_chunks" / "chunks.jsonl").write_text("{}\n", encoding="utf-8")

    report = migrate_course_corpus(cdir, apply=False)
    assert not report.applicable
    assert not report.already_migrated


# --------------------------------------------------------------------------- #
# Apply: flips everything, dual-read reader still resolves.                    #
# --------------------------------------------------------------------------- #

def test_apply_flips_sourceids_attrs_hashes_kind_and_dir(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_dart_course(root)
    old_top_sha = json.loads((cdir / "manifest.json").read_text())["dart_chunks_sha256"]

    report = migrate_course_corpus(cdir, apply=True)
    assert report.applied
    assert not report.rolled_back

    # Directory renamed dart_chunks -> semantik_chunks.
    assert not (cdir / "dart_chunks").exists()
    semantik = cdir / "semantik_chunks"
    assert semantik.is_dir()

    # chunks.jsonl: sourceIds + attrs flipped; no dart naming survives.
    chunks_text = (semantik / "chunks.jsonl").read_text(encoding="utf-8")
    assert '"sourceId": "semantik:' in chunks_text
    assert '"sourceId": "dart:' not in chunks_text
    assert "data-semantik-block-id" in chunks_text
    assert "data-dart-" not in chunks_text
    # The {slug}#{block_id} payload is intact; the value dart_converter preserved.
    assert "semantik:synthetic-dart-course#section-1" in chunks_text
    # html sits inside a JSON string (quotes escaped); the attr NAME flips but
    # the provenance VALUE dart_converter is preserved verbatim.
    assert "data-semantik-source" in chunks_text
    assert "dart_converter" in chunks_text

    # Sidecar: kind flipped, sha recomputed, provenance key renamed (value kept).
    sidecar = json.loads((semantik / "manifest.json").read_text(encoding="utf-8"))
    assert sidecar["chunkset_kind"] == "semantik"
    assert "source_dart_html_sha256" not in sidecar
    assert sidecar["source_semantik_html_sha256"] == "d" * 64
    disk_sha = hashlib.sha256((semantik / "chunks.jsonl").read_bytes()).hexdigest()
    assert sidecar["chunks_sha256"] == disk_sha
    assert report.new_chunks_sha256 == disk_sha

    # Archived HTML flipped.
    html = (cdir / "source" / "html" / "page01_accessible.html").read_text()
    assert "data-semantik-block-id" in html
    assert "data-dart-" not in html

    # Top-level manifest: key renamed to the fresh sha.
    top = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))
    assert "dart_chunks_sha256" not in top
    assert top["semantik_chunks_sha256"] == disk_sha
    assert disk_sha != old_top_sha  # sha changed because bodies changed

    # A backup dir was created and holds the pre-migration dart_chunks.
    assert report.backup_dir is not None and report.backup_dir.is_dir()
    bak_chunks = report.backup_dir / "dart_chunks" / "chunks.jsonl"
    assert '"sourceId": "dart:' in bak_chunks.read_text(encoding="utf-8")


def test_apply_result_is_dual_read_resolvable(tmp_path):
    """After migration the dual-read reader resolves the semantik_chunks/ dir."""
    root = tmp_path / "libv2"
    cdir = _make_dart_course(root)
    migrate_course_corpus(cdir, apply=True)

    resolved = resolve_imscc_chunks_path(cdir, "chunks.jsonl")
    assert resolved.parent.name == "semantik_chunks"
    assert resolved.is_file()
    text = resolved.read_text(encoding="utf-8")
    assert '"sourceId": "semantik:' in text


def test_apply_is_idempotent_second_run_noop(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_dart_course(root)
    migrate_course_corpus(cdir, apply=True)

    again = migrate_course_corpus(cdir, apply=True)
    assert again.already_migrated
    assert not again.applied


def test_apply_rolls_back_on_validation_failure(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_dart_course(root)
    before = _snapshot(cdir)

    class _FailValidator:
        valid = False
        errors = ["synthetic validation failure"]

    report = migrate_course_corpus(
        cdir,
        apply=True,
        validator=lambda c, r: _FailValidator(),
        repo_root=root,
    )
    assert not report.applied
    assert report.rolled_back
    assert report.validation_errors == ["synthetic validation failure"]

    # Every artifact restored: dart_chunks back, semantik_chunks gone, bytes match.
    assert (cdir / "dart_chunks").is_dir()
    assert not (cdir / "semantik_chunks").exists()
    after = _snapshot(cdir)
    # The backup dir is an extra path; compare only the original files.
    for rel, data in before.items():
        assert after.get(rel) == data, f"{rel} not restored"


def test_apply_refuses_when_semantik_dir_already_present(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_dart_course(root)
    (cdir / "semantik_chunks").mkdir()  # collision

    from LibV2.tools.libv2.migrate import MigrationError

    with pytest.raises(MigrationError) as exc:
        migrate_course_corpus(cdir, apply=True)
    assert exc.value.code == "SEMANTIK_DIR_EXISTS"
    # Rolled back: dart_chunks intact, still dart-named.
    assert (cdir / "dart_chunks" / "chunks.jsonl").is_file()
    assert '"sourceId": "dart:' in (cdir / "dart_chunks" / "chunks.jsonl").read_text()


# --------------------------------------------------------------------------- #
# Registry + driver                                                           #
# --------------------------------------------------------------------------- #

def test_dart_purge_registry_targets_1_1_with_corpus_transform():
    reg = build_dart_purge_registry()
    assert reg.target_version() == "1.1"
    step = reg.get("1.0")
    assert step is not None
    assert step.to_version == "1.1"
    assert step.corpus_transform is migrate_course_corpus


def test_default_registry_unchanged_still_1_0():
    """The dart-purge lives in its OWN registry; DEFAULT stays drift-locked."""
    from LibV2.tools.libv2.migrate import DEFAULT_REGISTRY

    assert DEFAULT_REGISTRY.target_version() == "1.0"


def test_plan_dart_purge_iterates_courses_without_writing(tmp_path):
    root = tmp_path / "libv2"
    a = _make_dart_course(root, "dart-a")
    b = _make_dart_course(root, "dart-b")
    before_a, before_b = _snapshot(a), _snapshot(b)

    reports = plan_dart_purge(root)
    slugs = {r.slug for r in reports}
    assert {"dart-a", "dart-b"} <= slugs
    assert all(not r.applied for r in reports)
    # Byte-identical: dry-run wrote nothing.
    assert _snapshot(a) == before_a
    assert _snapshot(b) == before_b

#!/usr/bin/env python3
"""Run the post-Courseforge tail directly on a passing-rewrite export.

This is the RELIABLE entry point for taking a Courseforge export that
already has a complete, passing rewrite (``03_content_development/*.html``
reconstructed from ``04_rewrite/blocks_final.jsonl``) all the way to an
ASKABLE course:

    package_imscc        -> <export>/05_final_package/<course>.imscc
    run_imscc_chunking   -> <libv2>/courses/<slug>/imscc_chunks/chunks.jsonl
    run_vector_indexing  -> <libv2>/courses/<slug>/vector_index/

It bypasses the full-workflow ``--resume`` path entirely (which is fragile:
it re-runs completed rewrites, persists stage-scoping that skips packaging,
and reconstructs empty content when blocks_final is missing). Instead it
calls the five registry handlers in sequence, threading each phase's
outputs to the next exactly as ``config/workflows.yaml``'s ``inputs_from``
declares.

NOTE on ``libv2_archival`` + ``finalization``: the full ``libv2_archival``
phase has three CRITICAL gates (``libv2_manifest``, ``packet_integrity_strict``,
``kg_quality_report``) that REQUIRE Trainforge-derived artifacts — a concept
graph, ``objectives.json``, semantik_chunks, and their sha256s — which a
Courseforge-only export does NOT have (archival ``depends_on`` trainforge
phases in the DAG). Those gates fail-closed on absent inputs BY DESIGN
(anti-silent-degradation). They are therefore NOT part of the "make the
course askable" minimal path: a queryable vector index only needs the
``imscc_chunks`` chunkset that ``run_imscc_chunking`` writes directly into the
course dir. Pass ``--archive`` to additionally attempt archival (it will
fail the Trainforge-gated criticals on a Courseforge-only export; useful
only after a full pipeline run that produced trainforge outputs).

Usage:
    python scripts/run_post_courseforge_tail.py \
        --project-id PROJ-<course-slug>-<timestamp> \
        --course-name <course-name> \
        --libv2-root /tmp/libv2-pkgtest
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _build_tool_registry,
    courseforge_exports_dir,
)


def _parse(envelope: str) -> dict:
    try:
        return json.loads(envelope)
    except (TypeError, ValueError):
        return {"success": False, "error": f"non-JSON envelope: {envelope!r}"}


async def run_tail(
    project_id: str,
    course_name: str,
    libv2_root: str | None,
    do_archive: bool,
    force_index: bool,
) -> int:
    reg = _build_tool_registry()
    export = courseforge_exports_dir() / project_id
    if not export.is_dir():
        print(f"FATAL: export not found: {export}", file=sys.stderr)
        return 2

    # Hard guard: the content dir must already carry real rewritten pages.
    content_dir = export / "03_content_development"
    htmls = sorted(content_dir.rglob("*.html"))
    if not htmls:
        print(
            f"FATAL: {content_dir} has no HTML pages — refusing to package "
            f"empty content. Reconstruct pages from blocks_final.jsonl first.",
            file=sys.stderr,
        )
        return 2
    total_bytes = sum(p.stat().st_size for p in htmls)
    print(f"[preflight] {len(htmls)} HTML pages, {total_bytes:,} bytes total")

    # ---- Phase 1: packaging ------------------------------------------------
    print("\n=== packaging (package_imscc) ===")
    pkg = _parse(await reg["package_imscc"](project_id=project_id))
    if not pkg.get("package_path") or pkg.get("error"):
        print(f"FATAL packaging: {json.dumps(pkg, indent=2)}", file=sys.stderr)
        return 3
    package_path = pkg["package_path"]
    size = Path(package_path).stat().st_size if Path(package_path).exists() else 0
    print(f"  package_path={package_path}")
    print(f"  size={size:,} bytes  html_modules={pkg.get('html_modules')}")

    # ---- Phase 2: imscc_chunking ------------------------------------------
    print("\n=== imscc_chunking (run_imscc_chunking) ===")
    chunk = _parse(
        await reg["run_imscc_chunking"](
            course_name=course_name,
            imscc_path=package_path,
            libv2_root=libv2_root,
        )
    )
    if not chunk.get("success"):
        print(f"FATAL chunking: {json.dumps(chunk, indent=2)}", file=sys.stderr)
        return 4
    imscc_chunks_path = chunk["imscc_chunks_path"]
    print(f"  imscc_chunks_path={imscc_chunks_path}")
    print(
        f"  chunks_count={chunk.get('chunks_count')}  "
        f"slug={chunk.get('course_slug')}  "
        f"sha256={chunk.get('imscc_chunks_sha256', '')[:16]}…"
    )

    # ---- Phase 2b: place the IMSCC where Studio's manifest reads it --------
    # The GUI course view (``/api/courses/<slug>/manifest``) parses lessons
    # from ``<libv2>/courses/<slug>/source/imscc/*.imscc`` (the convention
    # ``libv2_archival`` normally satisfies). Since we skip the full archival,
    # copy the package there so Studio renders all week pages instead of only
    # the Learning-Objectives map. Derived from imscc_chunks_path so it honors
    # whatever libv2 root the chunking step resolved.
    course_dir = Path(imscc_chunks_path).parent.parent  # <libv2>/courses/<slug>/
    imscc_dest_dir = course_dir / "source" / "imscc"
    imscc_dest_dir.mkdir(parents=True, exist_ok=True)
    imscc_dest = imscc_dest_dir / Path(package_path).name
    shutil.copy2(package_path, imscc_dest)
    print("\n=== place IMSCC for Studio display (source/imscc/) ===")
    print(f"  {imscc_dest}  ({imscc_dest.stat().st_size:,} bytes)")

    # ---- Phase 3 (optional): libv2_archival --------------------------------
    if do_archive:
        print("\n=== libv2_archival (archive_to_libv2) ===")
        arc = _parse(
            await reg["archive_to_libv2"](
                course_name=course_name,
                imscc_path=package_path,
                imscc_chunks_sha256=chunk.get("imscc_chunks_sha256"),
                libv2_root=libv2_root,
            )
        )
        print(f"  {json.dumps({k: arc.get(k) for k in ('course_slug', 'course_dir', 'manifest_path', 'error', 'error_code')}, indent=2)}")
        print(
            "  NOTE: archival's critical gates (libv2_manifest / "
            "packet_integrity / kg_quality) require Trainforge artifacts "
            "absent from a Courseforge-only export — expected to be gated."
        )

    # ---- Phase 4: vector_indexing -----------------------------------------
    print("\n=== vector_indexing (run_vector_indexing) ===")
    idx = _parse(
        await reg["run_vector_indexing"](
            course_name=course_name,
            chunkset="imscc",
            libv2_root=libv2_root,
            force=force_index,
        )
    )
    if not idx.get("success"):
        print(f"FATAL indexing: {json.dumps(idx, indent=2)}", file=sys.stderr)
        return 5
    print(f"  vector_index_dir={idx['vector_index_dir']}")
    print(
        f"  embedding_model_id={idx.get('embedding_model_id')}  "
        f"provider={idx.get('embedding_provider')}  dim={idx.get('embedding_dim')}"
    )
    print(
        f"  chunks_count={idx.get('chunks_count')}  "
        f"chunkset_kind={idx.get('chunkset_kind')}"
    )

    print("\n=== TAIL COMPLETE ===")
    print(json.dumps({
        "package_path": package_path,
        "package_size_bytes": size,
        "imscc_chunks_path": imscc_chunks_path,
        "vector_index_dir": idx["vector_index_dir"],
        "course_slug": idx.get("course_slug"),
    }, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-id", required=True)
    ap.add_argument(
        "--course-name",
        required=True,
        help="Slugifies to the LibV2 course dir. Use a DISTINCT slug to "
        "avoid clobbering an in-flight course (e.g. mycourse-pkgtest).",
    )
    ap.add_argument("--libv2-root", default=None)
    ap.add_argument("--archive", action="store_true", help="also try libv2_archival")
    ap.add_argument("--force-index", action="store_true")
    args = ap.parse_args()
    return asyncio.run(
        run_tail(
            args.project_id,
            args.course_name,
            args.libv2_root,
            args.archive,
            args.force_index,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

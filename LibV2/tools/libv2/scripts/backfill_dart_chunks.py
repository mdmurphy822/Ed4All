"""Phase 7c Subtask 18 — DART chunkset backfill operator script.

Operator-driven utility that scans existing LibV2 courses and emits a
``dart_chunks/`` chunkset for every course that is missing one. The
chunkset is the Phase 7b architectural contract:
``LibV2/courses/<slug>/dart_chunks/chunks.jsonl`` plus a sibling
``manifest.json`` validating against
``schemas/library/chunkset_manifest.schema.json``.

The script targets two operator surfaces:

1. **Bulk backfill** of legacy / pre-Phase-7b courses whose archive was
   produced before the ``chunking`` workflow phase landed. Those
   courses today carry only the legacy ``corpus/`` directory (or the
   imminent ``imscc_chunks/`` rename per Worker W15) and need a
   ``dart_chunks/`` chunkset minted from the staged DART HTML so
   Phase 7c's ``LibV2ManifestValidator`` extension (Subtask 17) doesn't
   fail closed on them.
2. **Targeted re-chunking** of a single course (``--course-slug``)
   when a chunker upgrade lands and an operator wants to refresh one
   archive without invoking the full ``textbook_to_course``
   workflow.

Design decisions:

- **Reuse over reimplementation.** The script delegates to
  ``MCP/tools/pipeline_tools.py::_run_dart_chunking`` (Phase 7b
  Subtask 11, commit ``5ccbf0c``) via the tool registry. That helper
  is the single source of truth for DART chunkset emit; reimplementing
  its parsing + ``Trainforge.chunker.chunk_content`` dispatch loop here
  would create a drift surface where two backends could disagree on
  the canonical chunkset shape. The helper is async, so we wrap the
  call site in ``asyncio.run`` (sync CLI, no event loop required).
- **Idempotency.** A course with an existing ``dart_chunks/manifest.json``
  is skipped by default. ``--force`` bypasses the skip.
- **Layout heuristics.** DART HTML on a real LibV2 course lives at
  ``<slug>/source/html/`` (per ``MCP/tools/pipeline_tools.py::5371-5378``
  and ``lib/validators/libv2_manifest.py::_EXPECTED_SUBDIRS``). The
  Phase 7b helper expects a ``staging_dir`` that contains the HTML
  files; we point it at ``<slug>/source/html/`` directly.
- **Fail-soft per course.** A chunker error (malformed HTML, parser
  exception, etc.) fails the affected course only — the script logs
  the error with the course slug and continues. The exit code is
  non-zero when any course fails so CI / cron callers see the
  regression.
- **Decision capture.** Per the Phase 7b/c spec, each successful
  backfill emits a ``decision_type="dart_chunks_backfill"`` event so
  the audit trail records WHO ran the backfill, WHEN, and against
  which course. Note: ``dart_chunks_backfill`` is not currently in
  ``schemas/events/decision_event.schema.json`` enum — under default
  (lenient) ``DECISION_VALIDATION_STRICT`` mode this emits a warning
  but does not fail. A schema-enum addition is a separate followup.

Usage::

    # Backfill every course missing dart_chunks/.
    python -m LibV2.tools.libv2.scripts.backfill_dart_chunks

    # Backfill a single course.
    python -m LibV2.tools.libv2.scripts.backfill_dart_chunks \\
        --course-slug rdf-shacl-551-2

    # Dry-run: log what would be done without writing anything.
    python -m LibV2.tools.libv2.scripts.backfill_dart_chunks --dry-run

    # Force re-chunk even if dart_chunks/ already exists.
    python -m LibV2.tools.libv2.scripts.backfill_dart_chunks \\
        --course-slug rdf-shacl-551-2 --force

Exit codes:
    0  success (or all targets skipped/dry-run).
    1  one or more courses failed during backfill.
    2  invalid invocation / missing libv2 root.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable when invoked as a script (not just
# via ``python -m``). Two parents up from the file location:
# .../LibV2/tools/libv2/scripts/backfill_dart_chunks.py -> repo root
# is four parents up.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


logger = logging.getLogger(__name__)


# Default LibV2 courses root, relative to the project root. Operators
# can override via ``--libv2-root`` to point at a checkout-of-checkouts
# topology or a sandboxed test fixture.
DEFAULT_LIBV2_COURSES_ROOT = _PROJECT_ROOT / "LibV2" / "courses"

# DART HTML lives at ``source/html/`` on archived courses (see
# ``MCP/tools/pipeline_tools.py::_archive_to_libv2`` line 5376 and
# ``lib/validators/libv2_manifest.py::_EXPECTED_SUBDIRS`` line 45).
# Older courses with non-standard layouts can be supported by passing
# ``--html-subdir`` directly (e.g. ``--html-subdir source/dart_html``).
DEFAULT_HTML_SUBDIR = "source/html"

# Sibling chunkset directory + manifest filename. Matches Phase 7b ST 11
# emit at ``MCP/tools/pipeline_tools.py:6649-6679``.
DART_CHUNKS_DIRNAME = "dart_chunks"
MANIFEST_FILENAME = "manifest.json"


@dataclass
class BackfillStats:
    """End-of-run summary counters."""

    backfilled: List[str] = field(default_factory=list)
    skipped_already_complete: List[str] = field(default_factory=list)
    skipped_no_html: List[str] = field(default_factory=list)
    failed: List[Dict[str, str]] = field(default_factory=list)
    dry_run: List[str] = field(default_factory=list)

    def total_attempted(self) -> int:
        return (
            len(self.backfilled)
            + len(self.skipped_already_complete)
            + len(self.skipped_no_html)
            + len(self.failed)
            + len(self.dry_run)
        )

    def has_failures(self) -> bool:
        return bool(self.failed)


def _enumerate_course_slugs(libv2_root: Path) -> List[str]:
    """Discover candidate course slugs under ``libv2_root``.

    A course directory is recognized by being a non-hidden,
    non-special subdirectory. Empty fixture directories (e.g.
    ``bogus-course-no-manifest``) are intentionally not filtered
    here — the per-course backfill loop handles the no-HTML case
    via :data:`BackfillStats.skipped_no_html` so the operator sees
    them in the summary.
    """
    if not libv2_root.exists() or not libv2_root.is_dir():
        return []
    slugs: List[str] = []
    for entry in sorted(libv2_root.iterdir()):
        if not entry.is_dir():
            continue
        # Skip hidden directories (e.g. ``.git`` if someone points at
        # the wrong root).
        if entry.name.startswith("."):
            continue
        # Skip the ``.gitkeep`` placeholder file (already filtered by
        # is_dir() but make the intent explicit).
        if entry.name == ".gitkeep":
            continue
        slugs.append(entry.name)
    return slugs


def _has_existing_chunkset(course_dir: Path) -> bool:
    """True when ``dart_chunks/manifest.json`` exists on disk.

    The presence check is deliberately shallow — we don't validate
    the manifest against the schema here. A malformed manifest is the
    Phase 7b ``ChunksetManifestValidator`` gate's responsibility; if
    an operator suspects a corrupt chunkset they can pass ``--force``
    to re-emit. The shallow check keeps this script's idempotency
    contract simple: "either there's a manifest or there isn't".
    """
    manifest_path = course_dir / DART_CHUNKS_DIRNAME / MANIFEST_FILENAME
    return manifest_path.is_file()


def _resolve_html_dir(course_dir: Path, html_subdir: str) -> Optional[Path]:
    """Resolve the directory of staged DART HTML for a course.

    Returns ``None`` when the directory doesn't exist or contains no
    ``*.html`` files. The chunker helper is fail-soft on empty input,
    but emitting a chunkset for a course with literally zero HTML
    pages is an operator surprise we'd rather flag than silently
    produce a zero-row ``chunks.jsonl``.
    """
    cand = course_dir / html_subdir
    if not cand.is_dir():
        return None
    # Quick sanity: at least one .html file under the candidate dir.
    # rglob handles nested figures-dir layouts where the HTML files
    # sit at the top level alongside ``{stem}_figures/``.
    for _ in cand.rglob("*.html"):
        return cand
    return None


async def _invoke_chunker(course_slug: str, html_dir: Path) -> Dict[str, Any]:
    """Dispatch ``_run_dart_chunking`` from the Phase 7b helper.

    We import inside the function to keep the script's import-time
    cost low (importing ``MCP.tools.pipeline_tools`` pulls in the
    full registry build path). The helper's contract is documented
    at ``MCP/tools/pipeline_tools.py:6362-6694``.

    Important: the helper hardcodes the chunkset destination to
    ``_PROJECT_ROOT / "LibV2" / "courses" / <slug>`` — it does not
    accept a libv2-root override (settled territory; not in this
    PR's scope). The script therefore always writes through the
    canonical project LibV2 first, then moves the chunkset to a
    custom ``--libv2-root`` afterward (handled in
    :func:`_backfill_one_course`). When ``--libv2-root`` matches the
    project's canonical ``LibV2/courses/`` path the move is a no-op.
    """
    # Lazy import — see docstring.
    from MCP.tools import pipeline_tools  # noqa: WPS433

    registry = pipeline_tools._build_tool_registry()  # noqa: SLF001
    run_dart_chunking = registry.get("run_dart_chunking")
    if run_dart_chunking is None:
        raise RuntimeError(
            "run_dart_chunking missing from MCP tool registry — "
            "expected Phase 7b ST 11 helper at "
            "MCP/tools/pipeline_tools.py::_run_dart_chunking"
        )

    # The helper accepts ``course_name`` (used as both course code and
    # slug source) and ``staging_dir``. We pass the course slug
    # uppercased as the ``course_name`` so chunk IDs carry a stable
    # course-code prefix matching Phase 7b ST 11's behavior:
    # ``course_code = course_name.upper().replace("-", "_")``.
    raw_response = await run_dart_chunking(
        course_name=course_slug,
        staging_dir=str(html_dir),
    )
    response = json.loads(raw_response)
    if not response.get("success"):
        raise RuntimeError(
            f"run_dart_chunking did not return success=true: {response!r}"
        )
    return response


def _canonical_libv2_courses_root() -> Path:
    """Return the canonical project-root ``LibV2/courses/`` path.

    Cached resolution against ``_PROJECT_ROOT`` so callers can
    compare a normalized ``--libv2-root`` argument against the path
    the helper actually writes to.
    """
    return (_PROJECT_ROOT / "LibV2" / "courses").resolve()


def _relocate_chunkset_if_needed(
    course_slug: str,
    libv2_root: Path,
    project_course_pre_existed: bool,
) -> None:
    """Move the just-emitted chunkset from project LibV2 to ``libv2_root``.

    The helper unconditionally writes to project LibV2; if the
    operator passed a different ``--libv2-root`` (typically a test
    fixture) we move the chunkset directory to the requested
    location, replacing any prior contents. When ``libv2_root``
    resolves to the project path this is a no-op.

    Idempotency: the destination directory is removed before the
    move so re-runs (with or without ``--force``) stay clean.

    When the project-LibV2 course shell did NOT exist before the
    helper ran (``project_course_pre_existed=False``), the relocate
    pass also removes the scaffolded sibling dirs the helper +
    decision-capture machinery created (``concept_graph/``,
    ``imscc_chunks/``, ``sources/``, etc. — see
    ``lib/libv2_storage.py``) so we don't leak empty fixture course
    directories into the operator's project tree. When the course
    pre-existed we leave the shell alone — those siblings hold real
    course data we must not touch.
    """
    canonical = _canonical_libv2_courses_root()
    target = libv2_root.resolve()
    if canonical == target:
        return
    src = canonical / course_slug / DART_CHUNKS_DIRNAME
    dst = target / course_slug / DART_CHUNKS_DIRNAME
    if not src.is_dir():
        # Nothing to move (helper failed mid-emit; the
        # _backfill_one_course caller will surface the error path).
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))

    if project_course_pre_existed:
        return

    # Course shell was minted by the helper invocation — clean it up
    # along with any sibling scaffolds the decision-capture /
    # libv2_storage machinery may have created. We only remove the
    # course directory when every remaining child is an empty
    # directory we recognise as scaffold output; ANY file (or any
    # nested file under a scaffold dir) defers cleanup to the
    # operator so we never silently delete unexpected content.
    src_course = canonical / course_slug
    if not src_course.is_dir():
        return
    # ``safe_to_clean`` is True iff every nested entry under
    # ``src_course`` is an empty directory — i.e. the helper's
    # scaffold tree has zero file content. Any single file aborts
    # the cleanup (defer to operator), even if it's deep in a
    # nested scaffold subdir.
    safe_to_clean = True
    try:
        for nested in src_course.rglob("*"):
            if nested.is_file():
                safe_to_clean = False
                break
    except OSError:
        safe_to_clean = False
    if safe_to_clean:
        try:
            shutil.rmtree(src_course)
        except OSError:
            # Race / readonly — leave it for ops review.
            pass


def _emit_decision_capture(
    course_slug: str,
    response: Dict[str, Any],
    operator: str,
) -> None:
    """Emit a ``dart_chunks_backfill`` decision-capture event.

    Operates in best-effort mode: a missing / unwriteable
    decision-capture surface logs a warning rather than failing the
    backfill. The capture is observability only — the canonical audit
    trail is the chunkset manifest itself.
    """
    try:
        from lib.decision_capture import DecisionCapture  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001 — capture is optional
        logger.debug(
            "decision-capture unavailable (%s); skipping backfill audit event",
            exc,
        )
        return

    try:
        capture = DecisionCapture(
            course_code=course_slug.upper().replace("-", "_"),
            phase="libv2-backfill",
            tool="libv2",
            streaming=False,
        )
    except Exception as exc:  # noqa: BLE001 — capture is optional
        logger.warning(
            "decision-capture init failed for %s (%s); skipping audit event",
            course_slug,
            exc,
        )
        return

    timestamp = datetime.now(timezone.utc).isoformat()
    chunks_count = response.get("chunks_count")
    chunker_version = response.get("chunker_version")
    chunks_sha256 = response.get("dart_chunks_sha256")
    rationale = (
        f"Backfilled DART chunkset for course '{course_slug}' on {timestamp} "
        f"by operator '{operator}'. Emitted {chunks_count} chunks via "
        f"Trainforge.chunker {chunker_version}; chunks_sha256={chunks_sha256!s}. "
        "Run scope: legacy / partially-migrated archive missing the Phase 7b "
        "dart_chunks/ chunkset (Subtask 18 operator script)."
    )
    try:
        capture.log_decision(
            decision_type="dart_chunks_backfill",
            decision=f"emit dart_chunks/ for {course_slug}",
            rationale=rationale,
            context=json.dumps(
                {
                    "course_slug": course_slug,
                    "operator": operator,
                    "chunks_count": chunks_count,
                    "chunker_version": chunker_version,
                    "chunks_sha256": chunks_sha256,
                    "timestamp": timestamp,
                },
                ensure_ascii=False,
            ),
        )
    except Exception as exc:  # noqa: BLE001 — capture is optional
        logger.warning(
            "decision-capture log failed for %s (%s); proceeding",
            course_slug,
            exc,
        )


def _backfill_one_course(
    course_slug: str,
    libv2_root: Path,
    html_subdir: str,
    dry_run: bool,
    force: bool,
    operator: str,
    stats: BackfillStats,
) -> None:
    """Run the backfill loop body for a single course slug."""
    course_dir = libv2_root / course_slug
    if not course_dir.is_dir():
        logger.error("Course directory missing: %s", course_dir)
        stats.failed.append(
            {
                "course_slug": course_slug,
                "error": "course directory does not exist",
            }
        )
        return

    chunkset_present = _has_existing_chunkset(course_dir)
    if chunkset_present and not force:
        logger.info(
            "[skip] %s — dart_chunks/ already present (pass --force to overwrite)",
            course_slug,
        )
        stats.skipped_already_complete.append(course_slug)
        return

    html_dir = _resolve_html_dir(course_dir, html_subdir)
    if html_dir is None:
        logger.info(
            "[skip] %s — no DART HTML files under %s/%s",
            course_slug,
            course_slug,
            html_subdir,
        )
        stats.skipped_no_html.append(course_slug)
        return

    if dry_run:
        logger.info(
            "[dry-run] would backfill %s from %s -> %s/%s/",
            course_slug,
            html_dir,
            course_slug,
            DART_CHUNKS_DIRNAME,
        )
        stats.dry_run.append(course_slug)
        return

    # When --force is set on a course that already has a chunkset,
    # remove the old directory before re-emitting so the new chunkset
    # doesn't carry stale sibling files (e.g. an extraneous old
    # chunks.jsonl with a different SHA from the manifest's).
    if chunkset_present and force:
        chunks_dir = course_dir / DART_CHUNKS_DIRNAME
        try:
            shutil.rmtree(chunks_dir)
            logger.info(
                "[force] removed existing %s/ before re-emit",
                chunks_dir.relative_to(libv2_root),
            )
        except OSError as exc:
            logger.error(
                "[fail] %s — could not remove existing chunkset (%s)",
                course_slug,
                exc,
            )
            stats.failed.append(
                {"course_slug": course_slug, "error": f"rmtree failed: {exc}"}
            )
            return

    # Capture whether the project-LibV2 course shell existed BEFORE
    # the helper ran. Threaded into the relocate pass so we can clean
    # up scaffold leakage when (and only when) we minted the shell
    # ourselves to satisfy the helper's hardcoded destination.
    canonical_libv2 = _canonical_libv2_courses_root()
    project_course_pre_existed = (canonical_libv2 / course_slug).is_dir()

    try:
        response = asyncio.run(_invoke_chunker(course_slug, html_dir))
    except Exception as exc:  # noqa: BLE001 — fail-soft per course
        logger.error("[fail] %s — chunker raised %s", course_slug, exc)
        stats.failed.append(
            {"course_slug": course_slug, "error": str(exc)}
        )
        return

    chunks_count = response.get("chunks_count", "?")
    chunker_version = response.get("chunker_version", "?")
    logger.info(
        "[ok]   %s — emitted %s chunks via Trainforge.chunker %s",
        course_slug,
        chunks_count,
        chunker_version,
    )
    stats.backfilled.append(course_slug)
    # Decision-capture BEFORE relocate: the LibV2Storage path used by
    # decision-capture re-materialises scaffold subdirs under the
    # project-LibV2 course shell on every instantiation. Emitting
    # the audit event first means the relocate pass can clean up
    # any post-emit scaffold leakage in one shot, instead of having
    # the leakage re-appear after we've already cleaned.
    _emit_decision_capture(course_slug, response, operator)

    # Move the chunkset from project LibV2 to ``--libv2-root`` when
    # they differ. The helper writes to a hardcoded path; the
    # operator script normalizes the destination for testability and
    # alternate-root deployment.
    try:
        _relocate_chunkset_if_needed(
            course_slug, libv2_root, project_course_pre_existed
        )
    except Exception as exc:  # noqa: BLE001 — relocate is best-effort
        logger.error(
            "[fail] %s — chunkset relocate to %s raised %s",
            course_slug,
            libv2_root,
            exc,
        )
        stats.failed.append(
            {"course_slug": course_slug, "error": f"relocate failed: {exc}"}
        )
        return


def _print_summary(stats: BackfillStats) -> None:
    """Print the end-of-run summary to stdout."""
    print()
    print("=" * 60)
    print("Backfill summary")
    print("=" * 60)
    print(f"  Total courses considered:   {stats.total_attempted()}")
    print(f"  Backfilled:                 {len(stats.backfilled)}")
    print(f"  Skipped (already complete): {len(stats.skipped_already_complete)}")
    print(f"  Skipped (no DART HTML):     {len(stats.skipped_no_html)}")
    print(f"  Dry-run plan only:          {len(stats.dry_run)}")
    print(f"  Failed:                     {len(stats.failed)}")
    if stats.backfilled:
        print()
        print("  Backfilled slugs:")
        for slug in stats.backfilled:
            print(f"    - {slug}")
    if stats.failed:
        print()
        print("  Failures:")
        for entry in stats.failed:
            print(f"    - {entry['course_slug']}: {entry['error']}")
    print("=" * 60)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Exposed as a module-level helper so the regression tests can
    introspect the supported flags without invoking ``main()``.
    """
    parser = argparse.ArgumentParser(
        prog="backfill_dart_chunks",
        description=(
            "Backfill DART chunksets (LibV2/courses/<slug>/dart_chunks/) "
            "for legacy / partially-migrated LibV2 courses. Phase 7c ST 18."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--libv2-root",
        type=Path,
        default=DEFAULT_LIBV2_COURSES_ROOT,
        help=(
            "Path to the LibV2 courses root (containing per-course "
            f"directories). Default: {DEFAULT_LIBV2_COURSES_ROOT}"
        ),
    )
    parser.add_argument(
        "--course-slug",
        type=str,
        default=None,
        help=(
            "Backfill exactly this slug (e.g. rdf-shacl-551-2). When "
            "omitted, scan every directory under --libv2-root."
        ),
    )
    parser.add_argument(
        "--html-subdir",
        type=str,
        default=DEFAULT_HTML_SUBDIR,
        help=(
            f"Subdirectory under each course holding the staged DART "
            f"HTML files. Default: {DEFAULT_HTML_SUBDIR}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what WOULD be done without writing anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-emit dart_chunks/ even when manifest.json already "
            "exists. Removes the existing chunkset directory first."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--operator",
        type=str,
        default=os.environ.get("USER", "unknown"),
        help=(
            "Operator identifier captured in the decision-capture "
            "event. Defaults to $USER."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Script entry point.

    Returns the integer exit code so the test suite can drive the
    function directly without going through ``sys.exit``.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    libv2_root: Path = args.libv2_root.resolve()
    if not libv2_root.exists() or not libv2_root.is_dir():
        logger.error(
            "LibV2 courses root not found or not a directory: %s",
            libv2_root,
        )
        return 2

    if args.course_slug:
        slugs = [args.course_slug]
    else:
        slugs = _enumerate_course_slugs(libv2_root)

    if not slugs:
        logger.warning("No course slugs to process under %s", libv2_root)
        _print_summary(BackfillStats())
        return 0

    logger.info(
        "Backfilling %d course(s) under %s (dry_run=%s, force=%s)",
        len(slugs),
        libv2_root,
        args.dry_run,
        args.force,
    )

    stats = BackfillStats()
    for slug in slugs:
        _backfill_one_course(
            course_slug=slug,
            libv2_root=libv2_root,
            html_subdir=args.html_subdir,
            dry_run=args.dry_run,
            force=args.force,
            operator=args.operator,
            stats=stats,
        )

    _print_summary(stats)
    return 1 if stats.has_failures() else 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())

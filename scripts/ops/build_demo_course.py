#!/usr/bin/env python3
"""Build the bundled Ed4All demo course from the tracked pipeline fixture.

B3 demo-course groundwork. This is the *pinned, documented* invocation that
mints the shippable demo bundle from the license-clean 3-page fixture PDF at
``tests/fixtures/pipeline/fixture_corpus.pdf`` (original synthetic content about
photosynthesis — see ``tests/fixtures/pipeline/build_fixture_pdf.py``). It wraps
``ed4all run textbook-to-course`` with a fixed course name + license +
attribution so the emitted ``course_manifest.json`` carries the B3
``license`` / ``attribution`` blocks and the archival step writes a
human-readable ``NOTICE`` file into the course dir.

The script REFUSES to run when the corpus fixture is missing (regenerate it via
``python tests/fixtures/pipeline/build_fixture_pdf.py``) and REFUSES a
``fake`` embedding provider in ``--full`` mode (a shipped bundle must carry a
real vector index — ``ED4ALL_EMBEDDING_ALLOW_FAKE`` gates fake indexes out of
production read paths).

Two build depths:

* default (retrieval-ready slice) — ``--stop-after imscc_chunking``. Stops
  before training synthesis + LibV2 archival; fast, produces the chunked
  course. NOTE: the ``license`` / ``attribution`` / ``NOTICE`` land at the
  ``libv2_archival`` phase, so this slice does NOT emit them — use ``--full``
  for the shippable bundle.
* ``--full`` — runs to completion (through ``libv2_archival`` +
  ``vector_indexing``) so the manifest carries the license/attribution blocks,
  the NOTICE is written, and a real vector index is built. This is the
  bundle you freeze + ship.

Usage::

    # Retrieval-ready slice (fast; no license/NOTICE emit):
    python scripts/ops/build_demo_course.py

    # Full shippable bundle (manifest license + NOTICE + real vector index):
    python scripts/ops/build_demo_course.py --full

    # Show the exact ed4all command without running it:
    python scripts/ops/build_demo_course.py --full --print-only

Full mint/freeze/ship runbook: ``docs/operations/demo-course.md``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------- #
# Pinned demo parameters (single source of truth for the bundle identity)
# ---------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PDF = REPO_ROOT / "tests" / "fixtures" / "pipeline" / "fixture_corpus.pdf"

# Course name → slug ``demo-photosynthesis`` via lib.ontology.slugs.libv2_course_slug.
DEMO_COURSE_NAME = "demo-photosynthesis"
DEMO_DOMAIN = "biology"

# The fixture is ORIGINAL synthetic content authored for this repo (no upstream
# rights holder), so it ships under a public-domain dedication with a plain
# attribution back to the fixture builder.
DEMO_LICENSE_NOTE = "CC0-1.0"
DEMO_ATTRIBUTION = (
    "Ed4All synthetic pipeline fixture — 'Introduction to Photosynthesis' "
    "(tests/fixtures/pipeline/build_fixture_pdf.py). Public-domain dedication."
)

# A shipped bundle must carry a REAL vector index; a fake-provider index is
# refused out of production read paths unless ED4ALL_EMBEDDING_ALLOW_FAKE=true.
_FAKE_EMBEDDING_PROVIDERS = {"fake"}


def _build_ed4all_command(*, full: bool) -> list[str]:
    """Assemble the pinned ``ed4all run`` argv for the demo build."""
    cmd = [
        sys.executable,
        "-m",
        "cli.main",
        "run",
        "textbook-to-course",
        "--corpus",
        str(FIXTURE_PDF),
        "--course-name",
        DEMO_COURSE_NAME,
        "--license-note",
        DEMO_LICENSE_NOTE,
        "--attribution",
        DEMO_ATTRIBUTION,
    ]
    if not full:
        # Retrieval-ready slice: stop before training synthesis + archival.
        cmd += ["--skip-training", "--stop-after", "imscc_chunking"]
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the bundled Ed4All demo course from the tracked "
        "pipeline fixture PDF.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the pipeline to completion (through libv2_archival + "
        "vector_indexing) so the manifest carries the license/attribution "
        "blocks, the NOTICE is written, and a real vector index is built. "
        "Without this flag the build stops after imscc_chunking (no "
        "license/NOTICE emit).",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the exact ed4all command and exit without running it.",
    )
    args = parser.parse_args(argv)

    # Refuse to run without the corpus fixture.
    if not FIXTURE_PDF.is_file():
        print(
            f"ERROR: demo corpus fixture is missing: {FIXTURE_PDF}\n"
            "Regenerate it with:\n"
            "  python tests/fixtures/pipeline/build_fixture_pdf.py",
            file=sys.stderr,
        )
        return 2

    # A shipped full bundle must not use a fake embedding provider.
    provider = os.environ.get("ED4ALL_EMBEDDING_PROVIDER", "st")
    if args.full and provider in _FAKE_EMBEDDING_PROVIDERS:
        print(
            f"ERROR: ED4ALL_EMBEDDING_PROVIDER={provider!r} builds a FAKE "
            "vector index, which is refused out of production read paths "
            "unless ED4ALL_EMBEDDING_ALLOW_FAKE=true. A shipped demo bundle "
            "must carry a real index — set ED4ALL_EMBEDDING_PROVIDER=st (or a "
            "real registry provider) and re-run.",
            file=sys.stderr,
        )
        return 2

    cmd = _build_ed4all_command(full=args.full)

    if args.print_only:
        # Shell-quote for copy-paste convenience.
        import shlex

        print(" ".join(shlex.quote(part) for part in cmd))
        return 0

    print(
        f"Building demo course {DEMO_COURSE_NAME!r} from {FIXTURE_PDF} "
        f"({'full bundle' if args.full else 'retrieval-ready slice'})...",
        file=sys.stderr,
    )
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

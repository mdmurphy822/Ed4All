"""Rebuild a course's `eval/holdout_split.json` with the current
HoldoutBuilder.

Run this after Wave 109 added `property_probes[]` to the holdout
schema; existing courses' splits were emitted before that change and
have an empty `property_probes` array, which silently SKIPs the
`min_per_property_accuracy` critical eval gate.

Usage:
    python -m Trainforge.eval.rebuild_holdout --course rdf-shacl-551-2
    python -m Trainforge.eval.rebuild_holdout --course-path LibV2/courses/rdf-shacl-551-2
    python -m Trainforge.eval.rebuild_holdout --course rdf-shacl-551-2 \\
        --holdout-pct 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.holdout_builder import HoldoutBuilder  # noqa: E402

logger = logging.getLogger(__name__)


def _resolve_course_path(course_slug: str) -> Path:
    """Map a course slug to the canonical `LibV2/courses/<slug>/` path."""
    return PROJECT_ROOT / "LibV2" / "courses" / course_slug


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a course's eval/holdout_split.json with the "
            "current HoldoutBuilder. Required after Wave 109 added "
            "property_probes[] to the schema."
        ),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--course",
        help=(
            "Course slug under LibV2/courses/, e.g. rdf-shacl-551-2. "
            "Resolves to <repo>/LibV2/courses/<slug>/."
        ),
    )
    src.add_argument(
        "--course-path",
        help=(
            "Absolute or relative path to the course directory "
            "(useful for CI / non-LibV2 layouts)."
        ),
    )
    parser.add_argument(
        "--holdout-pct", type=float, default=0.1,
        help="Fraction of edges per relation_type to withhold (default 0.1).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed; pinned in the output JSON (default 42).",
    )
    parser.add_argument(
        "--require-property-probes", action="store_true", default=True,
        help=(
            "Fail with non-zero exit if the rebuilt split has zero "
            "property_probes[] (default on; pass --no-require-property-probes "
            "to opt out for legacy courses without a property manifest)."
        ),
    )
    parser.add_argument(
        "--no-require-property-probes",
        dest="require_property_probes",
        action="store_false",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if args.course:
        course_path = _resolve_course_path(args.course)
    else:
        course_path = Path(args.course_path).resolve()

    if not course_path.exists():
        print(f"ERROR: course path does not exist: {course_path}",
              file=sys.stderr)
        return 2

    builder = HoldoutBuilder(
        course_path=course_path,
        holdout_pct=args.holdout_pct,
        seed=args.seed,
    )
    output_path = builder.build()
    print(f"Wrote {output_path}")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    counts = {
        "withheld_edges": len(payload.get("withheld_edges", [])),
        "probes": len(payload.get("probes", [])),
        "negative_probes": len(payload.get("negative_probes", [])),
        "property_probes": len(payload.get("property_probes", [])),
    }
    print("Counts:", counts)

    if args.require_property_probes and counts["property_probes"] == 0:
        print(
            "ERROR: rebuilt split has zero property_probes; the "
            "min_per_property_accuracy eval gate will SKIP. Either "
            "(a) author a property manifest at "
            "schemas/training/property_manifest.<family>.yaml, or "
            "(b) re-run with --no-require-property-probes for legacy "
            "courses without one.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

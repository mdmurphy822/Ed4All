"""Bloom-ladder initiative addendum AD-02 — DPO-yield projector.

CPU-only, zero-LLM famine guard: models a course's misconception-card DPO
yield BEFORE the multi-hour paraphrase loop spends any wall-clock. The July
famine failure class is that ELIGIBILITY is not ADMISSION — a corpus can
look synthesis-ready and still emit zero trainer-admissible preference
pairs, and the trainer only discovers a below-``min_dpo_pairs`` corpus
AFTER synthesis has already run (``dpo_fail_hard=true`` raises
``InsufficientPreferencePairsError`` post-hoc).

This script is a thin CLI wrapper over
:func:`lib.validators.dpo_yield_projection.project_dpo_yield` — the single
source of truth shared with the ``dpo_yield_projection`` warning gate wired
at ``textbook_to_course::training_synthesis``. All admission logic (chunk
misconception recovery, canonical-objective focus, Arm A/B eligibility,
trainer admissibility) is imported from the real predicates that
``training_synthesis`` itself calls — nothing here re-implements them.

Usage::

    python -m Trainforge.scripts.model_dpo_yield --course-dir <dir>
    python -m Trainforge.scripts.model_dpo_yield --course-dir <dir> \\
        --objectives-path <path> --min-dpo-pairs 80 --json

``--course-dir`` accepts either a real LibV2 course dir (chunks resolved via
``lib.libv2_storage.resolve_imscc_chunks_path``, which prefers
``imscc_chunks/`` and falls back through the pre-rename chunkset dir /
legacy
``corpus/``) or a synthetic fixture dir carrying ``imscc_chunks/chunks.jsonl``
directly.

Exit codes:
    0  projected_admissible >= min_dpo_pairs.
    1  projected_admissible < min_dpo_pairs (the famine signal).
    2  required input (chunks / objectives) could not be resolved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.validators.dpo_yield_projection import (  # noqa: E402
    DEFAULT_MIN_DPO_PAIRS,
    load_canonical_objectives,
    project_dpo_yield,
    read_chunks_jsonl,
)

#: Search chain for the canonical objectives artifact, mirroring
#: ``Trainforge/synthesize_training.py``'s ``_objectives_candidates`` — the
#: same locations the real ``training_synthesis`` phase searches, in the
#: same precedence order, so this script's projection sees the same
#: objectives a real run would.
_OBJECTIVES_CANDIDATE_SUFFIXES = (
    Path("course_planning") / "synthesized_objectives.json",
    Path("objectives.json"),
    Path("synthesized_objectives.json"),
    Path("01_learning_objectives") / "synthesized_objectives.json",
)


def _resolve_chunks_path(course_dir: Path) -> Optional[Path]:
    from lib.libv2_storage import resolve_imscc_chunks_path

    candidate = resolve_imscc_chunks_path(course_dir, "chunks.jsonl")
    if candidate.exists():
        return candidate
    return None


def _resolve_objectives_path(
    course_dir: Path, explicit: Optional[str],
) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    for suffix in _OBJECTIVES_CANDIDATE_SUFFIXES:
        candidate = course_dir / suffix
        if candidate.exists():
            return candidate
    return None


def _render_table(report: Dict[str, Any], *, course_dir: Path) -> str:
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append("DPO YIELD PROJECTION")
    lines.append("=" * 64)
    lines.append(f"  course_dir             : {course_dir}")
    lines.append(f"  chunks walked          : {report['chunk_count']}")
    lines.append(f"  misconception cards    : {report['cards_found']}")
    lines.append("  cards by rung:")
    if report["cards_by_rung"]:
        for rung, count in report["cards_by_rung"].items():
            lines.append(f"    {rung:<12}: {count}")
    else:
        lines.append("    (none)")
    lines.append(f"  arm_a_admitted         : {report['arm_a_admitted']}")
    lines.append(f"  projected_admissible   : {report['projected_admissible']}")
    lines.append("  projected_admissible by rung:")
    if report["projected_admissible_by_rung"]:
        for rung, count in report["projected_admissible_by_rung"].items():
            lines.append(f"    {rung:<12}: {count}")
    else:
        lines.append("    (none)")
    lines.append(f"  min_dpo_pairs          : {report['min_dpo_pairs']}")
    lines.append(f"  shortfall              : {report['shortfall']}")
    verdict = "FAMINE (below min_dpo_pairs)" if report["deficit"] else "OK"
    lines.append(f"  verdict                : {verdict}")
    lines.append("=" * 64)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_dpo_yield",
        description=(
            "Project per-course DPO-admissible preference-pair yield "
            "before running the training-synthesis paraphrase loop."
        ),
    )
    parser.add_argument(
        "--course-dir",
        required=True,
        help=(
            "LibV2 course dir, or a fixture dir carrying "
            "imscc_chunks/chunks.jsonl directly."
        ),
    )
    parser.add_argument(
        "--objectives-path",
        default=None,
        help=(
            "Explicit canonical objectives JSON path. Defaults to the same "
            "search chain training_synthesis uses under --course-dir."
        ),
    )
    parser.add_argument(
        "--min-dpo-pairs",
        type=int,
        default=DEFAULT_MIN_DPO_PAIRS,
        help=f"Trainer floor to project against (default {DEFAULT_MIN_DPO_PAIRS}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-parseable JSON instead of the human table.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    course_dir = Path(args.course_dir)

    chunks_path = _resolve_chunks_path(course_dir)
    if chunks_path is None:
        message = f"no chunks.jsonl resolvable under {course_dir}"
        if args.json:
            print(json.dumps({"error": message}))
        else:
            print(f"ERROR: {message}", file=sys.stderr)
        return 2

    objectives_path = _resolve_objectives_path(course_dir, args.objectives_path)
    if objectives_path is None:
        message = (
            f"no canonical objectives artifact resolvable under {course_dir} "
            "(searched course_planning/synthesized_objectives.json, "
            "objectives.json, synthesized_objectives.json, "
            "01_learning_objectives/synthesized_objectives.json; pass "
            "--objectives-path to point at one explicitly)"
        )
        if args.json:
            print(json.dumps({"error": message}))
        else:
            print(f"ERROR: {message}", file=sys.stderr)
        return 2

    chunks = read_chunks_jsonl(chunks_path)
    objectives = load_canonical_objectives(objectives_path)

    report = project_dpo_yield(
        chunks, objectives, min_dpo_pairs=args.min_dpo_pairs,
    )
    report["chunks_path"] = str(chunks_path)
    report["objectives_path"] = str(objectives_path)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_table(report, course_dir=course_dir))

    return 1 if report["deficit"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

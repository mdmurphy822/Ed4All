"""Wave 137d-3 — inspect form_data coverage checkpoint history.

Operator workflow::

    python -m Trainforge.scripts.show_form_data_coverage \\
        --course-code rdf-shacl-551-2

Default: prints the latest checkpoint row as a column-aligned table.
``--all`` walks the full history. ``--format json`` emits a
machine-parseable dump (the latest row when ``--all`` is unset; the
full history array when ``--all`` is set).

Exit codes:
    0  success — at least one row rendered.
    2  checkpoint file absent or empty (no eval has run since
       Wave 137d-2 landed for this course).
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


def _resolve_default_checkpoint_path(course_code: str) -> Path:
    return (
        PROJECT_ROOT
        / "LibV2"
        / "courses"
        / course_code
        / "eval"
        / "form_data_coverage_checkpoint.jsonl"
    )


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_int(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _render_table(row: Dict[str, Any]) -> str:
    """Render one checkpoint row as a column-aligned operator table."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("FORM_DATA COVERAGE CHECKPOINT")
    lines.append("=" * 60)
    lines.append(f"  timestamp           : {row.get('timestamp', 'n/a')}")
    lines.append(f"  course_slug         : {row.get('course_slug', 'n/a')}")
    lines.append(f"  model_id            : {row.get('model_id', 'n/a')}")
    lines.append(f"  family              : {row.get('family', 'n/a')}")
    lines.append(
        f"  manifest_coverage   : {_format_pct(row.get('manifest_coverage_pct'))}"
    )
    lines.append(
        f"  complete_count      : {_format_int(row.get('complete_count'))}"
    )
    lines.append(
        f"  degraded_count      : {_format_int(row.get('degraded_count'))}"
    )
    lines.append(
        f"  promotion_decision  : {row.get('promotion_decision', 'n/a')}"
    )
    block_reasons = row.get("promotion_block_reasons") or []
    if block_reasons:
        lines.append("  block_reasons       :")
        for code in block_reasons:
            lines.append(f"    - {code}")
    else:
        lines.append("  block_reasons       : (none)")

    family_map = row.get("family_coverage_map") or {}
    if family_map:
        lines.append("")
        lines.append("  family_coverage:")
        for fam_name in sorted(family_map.keys()):
            entry = family_map[fam_name] or {}
            complete = entry.get("complete", 0)
            total = entry.get("total", 0)
            status = entry.get("status", "n/a")
            lines.append(
                f"    {fam_name:<24} {complete}/{total}  [{status}]"
            )
    lines.append("=" * 60)
    return "\n".join(lines)


def _compute_live_row(course_code: str) -> Optional[dict]:
    """Wave 137 follow-up: compute a live coverage snapshot when the
    checkpoint history is absent. Returns a dict in the same shape as
    a checkpoint row (timestamp + coverage fields + live=True marker)
    so the renderer + --format json work identically."""
    from datetime import datetime, timezone

    try:
        from lib.ontology.property_manifest import load_property_manifest
        from lib.validators.form_data_coverage import compute_coverage_metrics
    except ImportError:
        return None

    try:
        manifest = load_property_manifest(course_code)
    except FileNotFoundError:
        return None

    try:
        metrics = compute_coverage_metrics(manifest.family, manifest=manifest)
    except Exception:
        return None

    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": None,
        "course_slug": course_code,
        "family": manifest.family,
        "live": True,
        **metrics,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="show_form_data_coverage",
        description=(
            "Inspect the FORM_DATA coverage checkpoint history for a "
            "LibV2 course. Default: latest row as a column-aligned "
            "table. --all walks history; --format json emits machine-"
            "parseable output."
        ),
    )
    parser.add_argument(
        "--course-code",
        required=True,
        help="LibV2 course slug (e.g., rdf-shacl-551-2).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show every row in the checkpoint history (default: latest only).",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format. Default: table.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help=(
            "Override the checkpoint file path. Default: "
            "LibV2/courses/<course-code>/eval/form_data_coverage_checkpoint.jsonl."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    explicit_path = args.checkpoint_path is not None
    if args.checkpoint_path:
        path = Path(args.checkpoint_path)
    else:
        path = _resolve_default_checkpoint_path(args.course_code)

    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        if explicit_path:
            # Operator named a specific path; refuse silent fallback.
            print(
                "checkpoint not found or empty: " + str(path),
                file=sys.stderr,
            )
            return 2
        # Wave 137 follow-up: fall back to a live snapshot when the default
        # checkpoint is missing (no eval has run since Wave 137d-2 landed).
        # Lets operators inspect baseline coverage BEFORE the first retrain.
        live_row = _compute_live_row(args.course_code)
        if live_row is None:
            print(
                "checkpoint not found and could not compute live coverage "
                f"(no manifest resolvable for course '{args.course_code}'): "
                + str(path),
                file=sys.stderr,
            )
            return 2
        if args.format == "json":
            print(json.dumps(live_row, indent=2))
        else:
            print("(live snapshot — no checkpoint history yet)")
            print(_render_table(live_row))
        return 0

    raw = path.read_text(encoding="utf-8")
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if not rows:
        print("checkpoint file is empty", file=sys.stderr)
        return 2

    selected = rows if args.all else [rows[-1]]

    if args.format == "json":
        if args.all:
            print(json.dumps(selected, indent=2))
        else:
            print(json.dumps(selected[0], indent=2))
        return 0

    # Table format.
    for i, row in enumerate(selected):
        if i > 0:
            print("")
        print(_render_table(row))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

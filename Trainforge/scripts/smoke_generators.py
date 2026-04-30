"""Smoke-test all four deterministic training-data generators.

Wave 124 follow-up. Operator-facing CLI for verifying that all four
audit-anchored generators (kg_metadata, violation, abstention,
schema_translation) wire correctly against a real LibV2 course and
produce schema-valid pairs in <30s without an LLM call.

Use before committing to a full corpus rebuild: it confirms the
generator surfaces are healthy on the real graph + manifest. Mirrors
the `Trainforge/eval/slm_eval_harness.py --smoke` pattern: small N,
sidecar output, schema-validate every emit, exit 0 on success / 1 on
any failure.

Usage::

    python -m Trainforge.scripts.smoke_generators \
      --course-code rdf-shacl-551-2 \
      --max-pairs-per-generator 5

Exit code 0 means: all four generators emitted >= 1 schema-valid pair.
Exit code 1 means: at least one generator failed; the failing
generator + first error is printed to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.decision_capture import DecisionCapture  # noqa: E402
from lib.ontology.property_manifest import load_property_manifest  # noqa: E402
from Trainforge.generators.abstention_generator import (  # noqa: E402
    generate_abstention_pairs,
)
from Trainforge.generators.kg_metadata_generator import (  # noqa: E402
    generate_kg_metadata_pairs,
)
from Trainforge.generators.schema_translation_generator import (  # noqa: E402
    generate_schema_translation_pairs,
)
from Trainforge.generators.violation_generator import (  # noqa: E402
    generate_violation_pairs,
)

import jsonschema  # noqa: E402

logger = logging.getLogger("smoke_generators")


def _resolve_pedagogy_graph(course_dir: Path) -> Dict[str, Any]:
    candidates = [
        course_dir / "graph" / "pedagogy_graph.json",
        course_dir / "pedagogy" / "pedagogy_graph.json",
        course_dir / "pedagogy_graph.json",
    ]
    for p in candidates:
        if p.is_file():
            return json.loads(p.read_text())
    raise FileNotFoundError(
        f"pedagogy_graph.json not found under {course_dir} "
        f"(checked: graph/, pedagogy/, root)"
    )


def _load_pair_schema() -> Dict[str, Any]:
    schema_path = (
        PROJECT_ROOT / "schemas" / "knowledge" / "instruction_pair.schema.json"
    )
    return json.loads(schema_path.read_text())


def _validate_pairs(
    pairs: List[Dict[str, Any]],
    schema: Dict[str, Any],
    label: str,
) -> Tuple[int, List[str]]:
    errors: List[str] = []
    valid = 0
    for i, p in enumerate(pairs):
        try:
            jsonschema.validate(p, schema)
            valid += 1
        except jsonschema.ValidationError as e:
            errors.append(f"{label}[{i}]: {e.message[:200]}")
            if len(errors) >= 3:
                errors.append(f"{label}: ... ({len(pairs) - i - 1} more pairs not checked)")
                break
    return valid, errors


def _run_generator(
    name: str,
    fn,
    capture_kwargs: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    capture = DecisionCapture(
        course_code="SMOKE_TEST",
        phase="trainforge-training",
        tool="trainforge",
        streaming=False,
    )
    try:
        pairs, stats = fn(capture=capture, **capture_kwargs)
    except Exception as e:
        return {
            "generator": name,
            "ok": False,
            "pair_count": 0,
            "schema_valid_count": 0,
            "errors": [f"{type(e).__name__}: {e}"],
            "trace": traceback.format_exc(),
            "sample": None,
        }
    valid, errors = _validate_pairs(pairs, schema, name)
    sample = pairs[0] if pairs else None
    return {
        "generator": name,
        "ok": len(pairs) >= 1 and valid == len(pairs) and not errors,
        "pair_count": len(pairs),
        "schema_valid_count": valid,
        "errors": errors,
        "stats": _stats_to_dict(stats),
        "sample": sample,
    }


def _stats_to_dict(stats: Any) -> Dict[str, Any]:
    if hasattr(stats, "__dict__"):
        return {k: v for k, v in stats.__dict__.items() if not k.startswith("_")}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test all four deterministic generators.",
    )
    parser.add_argument(
        "--course-code",
        default="rdf-shacl-551-2",
        help="LibV2 course slug (default: rdf-shacl-551-2).",
    )
    parser.add_argument(
        "--max-pairs-per-generator",
        type=int,
        default=5,
        help="Cap each generator's emit (default: 5).",
    )
    parser.add_argument(
        "--libv2-root",
        type=Path,
        default=PROJECT_ROOT / "LibV2" / "courses",
        help="LibV2 courses root (default: LibV2/courses).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-parseable JSON summary instead of human table.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s: %(levelname)s: %(message)s",
    )

    course_dir = args.libv2_root / args.course_code
    if not course_dir.is_dir():
        print(
            f"FATAL: course directory not found: {course_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        pedagogy_graph = _resolve_pedagogy_graph(course_dir)
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1

    try:
        manifest = load_property_manifest(args.course_code)
    except Exception as e:
        print(
            f"FATAL: load_property_manifest({args.course_code!r}) failed: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1

    schema = _load_pair_schema()
    cap = int(args.max_pairs_per_generator)

    results: List[Dict[str, Any]] = []
    results.append(
        _run_generator(
            "kg_metadata",
            generate_kg_metadata_pairs,
            {
                "pedagogy_graph": pedagogy_graph,
                "max_pairs": cap,
                "negatives_per_positive": 1,
                "seed": 17,
            },
            schema,
        )
    )
    results.append(
        _run_generator(
            "violation",
            generate_violation_pairs,
            {"seed": 17, "max_pairs": cap},
            schema,
        )
    )
    results.append(
        _run_generator(
            "abstention",
            generate_abstention_pairs,
            {
                "pedagogy_graph": pedagogy_graph,
                "max_pairs": cap,
                "silent_per_chunk": 1,
                "seed": 17,
            },
            schema,
        )
    )
    results.append(
        _run_generator(
            "schema_translation",
            generate_schema_translation_pairs,
            {
                "manifest": manifest,
                "max_pairs": cap,
                "seed": 17,
            },
            schema,
        )
    )

    overall_ok = all(r["ok"] for r in results)

    if args.json:
        summary = {
            "course_code": args.course_code,
            "max_pairs_per_generator": cap,
            "generators": [
                {k: v for k, v in r.items() if k != "trace"} for r in results
            ],
            "overall_ok": overall_ok,
        }
        print(json.dumps(summary, indent=2))
    else:
        print(f"Smoke test: {args.course_code} (cap={cap} pairs/generator)")
        print(
            f"{'GENERATOR':<22} {'PAIRS':>7} {'VALID':>7} {'STATUS':>10}"
        )
        print("-" * 50)
        for r in results:
            status = "OK" if r["ok"] else "FAIL"
            print(
                f"{r['generator']:<22} "
                f"{r['pair_count']:>7} "
                f"{r['schema_valid_count']:>7} "
                f"{status:>10}"
            )
            if r["errors"]:
                for err in r["errors"]:
                    print(f"    error: {err}")
            if r["sample"] and overall_ok:
                prompt_preview = str(r["sample"].get("prompt", ""))[:80]
                print(f"    sample prompt: {prompt_preview}")
        print("-" * 50)
        print(f"Overall: {'PASS' if overall_ok else 'FAIL'}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

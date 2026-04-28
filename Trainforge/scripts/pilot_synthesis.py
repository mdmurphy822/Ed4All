"""Wave 109 / Phase C — pilot synthesis CLI.

Runs a small-N synthesis pass + emits a property-coverage Markdown
report so an operator can validate paraphrase quality BEFORE
committing to a full-corpus rebuild. Does not train.

Wave 117: report-formatting helpers moved to
``Trainforge.scripts.pilot_report_helpers`` so the in-flight writer
inside ``synthesize_training.run_synthesis`` can share one
implementation. Behavior of this CLI is unchanged.

Example:

    python -m Trainforge.scripts.pilot_synthesis \\
        --corpus LibV2/courses/<course-slug> \\
        --course-code <course-slug> \\
        --provider mock \\
        --max-pairs 50

(Use ``--provider claude_session`` only when invoked through the
workflow runner / MCP tool — that path injects the LocalDispatcher
the provider requires.)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.synthesize_training import run_synthesis  # noqa: E402
from Trainforge.scripts.pilot_report_helpers import (  # noqa: E402
    count_property_coverage_from_jsonl,
    format_pilot_report,
    template_distribution_from_jsonl,
    write_pilot_report_atomic,
)
from lib.ontology.property_manifest import (  # noqa: E402
    load_property_manifest,
)

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pilot synthesis quality check")
    parser.add_argument("--corpus", required=True, help="Course output dir.")
    parser.add_argument("--course-code", required=True)
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "anthropic", "claude_session", "together", "local"],
    )
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpus = Path(args.corpus).resolve()
    if not corpus.exists():
        logger.error("Corpus dir does not exist: %s", corpus)
        return 2

    try:
        manifest = load_property_manifest(args.course_code)
    except FileNotFoundError as exc:
        logger.error(
            "No property manifest for course '%s': %s", args.course_code, exc,
        )
        return 2

    logger.info(
        "Pilot synthesis: corpus=%s provider=%s max_pairs=%d",
        corpus, args.provider, args.max_pairs,
    )
    stats = run_synthesis(
        corpus_dir=corpus,
        course_code=args.course_code,
        provider=args.provider,
        seed=args.seed,
        max_pairs=args.max_pairs,
    )

    inst_path = corpus / "training_specs" / "instruction_pairs.jsonl"
    counts = count_property_coverage_from_jsonl(inst_path, manifest)
    templates = template_distribution_from_jsonl(inst_path)
    # Wave 117: pilot_synthesis.py is the post-hoc surface; the report
    # represents the final state, not a mid-run snapshot. We pass the
    # eligible-chunk count as both ``processed`` and ``total`` because
    # the CLI runs synthesis to completion before formatting.
    chunks_count = stats.chunks_eligible
    report = format_pilot_report(
        course_slug=args.course_code,
        provider=args.provider,
        counts=counts,
        manifest=manifest,
        templates=templates,
        total_pairs=stats.instruction_pairs_emitted,
        chunks_processed=chunks_count,
        chunks_total=chunks_count,
        in_flight=False,
    )
    report_path = corpus / "training_specs" / "pilot_report.md"
    write_pilot_report_atomic(report_path, report)
    logger.info("Pilot report written to %s", report_path)
    print(report)
    failures = sum(
        1 for prop in manifest.properties
        if counts.get(prop.id, 0) < prop.min_pairs
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

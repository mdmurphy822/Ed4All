"""Wave 109 / Phase C — pilot synthesis CLI.

Runs a small-N synthesis pass + emits a property-coverage Markdown
report so an operator can validate paraphrase quality BEFORE
committing to a full-corpus rebuild. Does not train.

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
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.synthesize_training import run_synthesis  # noqa: E402
from lib.ontology.property_manifest import (  # noqa: E402
    PropertyManifest,
    load_property_manifest,
)

logger = logging.getLogger(__name__)


def _count_property_coverage(
    inst_path: Path, manifest: PropertyManifest,
) -> Dict[str, int]:
    counts = {p.id: 0 for p in manifest.properties}
    if not inst_path.exists():
        return counts
    with inst_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = f"{row.get('prompt', '')} {row.get('completion', '')}"
            for prop in manifest.properties:
                if prop.matches(text):
                    counts[prop.id] += 1
    return counts


def _template_distribution(inst_path: Path) -> Counter:
    c: Counter = Counter()
    if not inst_path.exists():
        return c
    with inst_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            c[str(row.get("template_id") or "<none>")] += 1
    return c


def _format_report(
    course_slug: str,
    provider: str,
    max_pairs: int,
    counts: Dict[str, int],
    manifest: PropertyManifest,
    templates: Counter,
    total_pairs: int,
) -> str:
    lines: List[str] = []
    lines.append(f"# Pilot synthesis report — {course_slug}\n")
    lines.append(f"- **Provider:** `{provider}`")
    lines.append(f"- **Max pairs:** {max_pairs}")
    lines.append(f"- **Total emitted pairs:** {total_pairs}\n")
    lines.append("## Property coverage\n")
    lines.append("| Property | Pairs | Floor | Status |")
    lines.append("|---|---|---|---|")
    failures = 0
    for prop in manifest.properties:
        seen = counts.get(prop.id, 0)
        ok = seen >= prop.min_pairs
        if not ok:
            failures += 1
        status = "PASS" if ok else "FAIL"
        lines.append(f"| `{prop.curie}` | {seen} | {prop.min_pairs} | {status} |")
    lines.append("")
    lines.append(f"**Failures:** {failures} / {len(manifest.properties)}\n")
    lines.append("## Top 10 templates\n")
    lines.append("| Template | Count |")
    lines.append("|---|---|")
    for template, count in templates.most_common(10):
        lines.append(f"| `{template}` | {count} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Pilot synthesis quality check")
    parser.add_argument("--corpus", required=True, help="Course output dir.")
    parser.add_argument("--course-code", required=True)
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "anthropic", "claude_session", "together"],
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
    counts = _count_property_coverage(inst_path, manifest)
    templates = _template_distribution(inst_path)
    report = _format_report(
        course_slug=args.course_code,
        provider=args.provider,
        max_pairs=args.max_pairs,
        counts=counts,
        manifest=manifest,
        templates=templates,
        total_pairs=stats.instruction_pairs_emitted,
    )
    report_path = corpus / "training_specs" / "pilot_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info("Pilot report written to %s", report_path)
    print(report)
    failures = sum(
        1 for prop in manifest.properties
        if counts.get(prop.id, 0) < prop.min_pairs
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""Wave 117: shared pilot-report helpers.

Used by both ``Trainforge.scripts.pilot_synthesis`` (post-hoc, reads
JSONL from disk) and ``Trainforge.synthesize_training`` (in-flight,
operates on in-memory record lists). Same report shape; the JSONL
readers are thin wrappers over the in-memory variants.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

# Avoid a hard dep on PropertyManifest at import time so Trainforge
# code paths that don't use the helpers don't pay the import cost.
# Type-only via TYPE_CHECKING is fine here.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lib.ontology.property_manifest import PropertyManifest


def count_property_coverage_from_records(
    records: Iterable[Mapping[str, object]],
    manifest: "PropertyManifest",
) -> Dict[str, int]:
    """Count instruction pairs whose prompt+completion text matches each
    declared property's surface forms. Operates on an in-memory record
    list so it can run mid-synthesis."""
    counts = {p.id: 0 for p in manifest.properties}
    for row in records:
        text = f"{row.get('prompt', '')} {row.get('completion', '')}"
        for prop in manifest.properties:
            if prop.matches(text):
                counts[prop.id] += 1
    return counts


def template_distribution_from_records(
    records: Iterable[Mapping[str, object]],
) -> Counter:
    """Count pairs by template_id for diversity reporting."""
    c: Counter = Counter()
    for row in records:
        c[str(row.get("template_id") or "<none>")] += 1
    return c


def format_pilot_report(
    *,
    course_slug: str,
    provider: str,
    counts: Dict[str, int],
    manifest: "PropertyManifest",
    templates: Counter,
    total_pairs: int,
    chunks_processed: int,
    chunks_total: int,
    in_flight: bool,
    capped_at_max_pairs: bool = False,
    max_pairs_cap: int | None = None,
) -> str:
    """Render the markdown report. ``in_flight=True`` adds a banner
    indicating the report is mid-run; ``in_flight=False`` reads as the
    final post-run snapshot.

    Wave 119: when ``capped_at_max_pairs=True``, prepend a loud warning
    banner so an operator opening the report can't miss that property
    floors are evaluated against a clipped run (the failure mode that
    bit Wave 118's first 14B rerun).
    """
    lines: List[str] = []
    lines.append(f"# Pilot synthesis report — {course_slug}\n")
    if capped_at_max_pairs:
        cap_hint = f" (cap={max_pairs_cap})" if max_pairs_cap is not None else ""
        lines.append(
            f"> **WARNING — run capped at `--max-pairs`{cap_hint}.** Property "
            f"coverage below is evaluated against a clipped run; properties "
            f"whose surface forms appear later in the corpus may show 0 pairs "
            f"purely because their chunks were never visited. Re-run without "
            f"`--max-pairs` (or with a cap above eligible-chunks) before "
            f"interpreting the floor results.\n"
        )
    if in_flight:
        lines.append(
            f"> **In-flight snapshot** — chunks {chunks_processed}/{chunks_total} "
            f"processed ({100 * chunks_processed / max(1, chunks_total):.1f}%). "
            f"This report is regenerated periodically during the run; the "
            f"final post-run pass overwrites it once.\n"
        )
    lines.append(f"- **Provider:** `{provider}`")
    lines.append(f"- **Chunks processed:** {chunks_processed} / {chunks_total}")
    lines.append(f"- **Total emitted instruction pairs:** {total_pairs}\n")
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


def write_pilot_report_atomic(report_path: Path, content: str) -> None:
    """Write the report to ``report_path`` via tmp-and-rename so a
    concurrent ``cat`` / ``less`` doesn't observe a half-written file.
    The ``tail -f`` operator pattern is safe — atomic rename means the
    file always reflects a complete report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(report_path)


# JSONL-reading wrappers retained for pilot_synthesis.py's post-hoc use.
def count_property_coverage_from_jsonl(
    inst_path: Path, manifest: "PropertyManifest",
) -> Dict[str, int]:
    if not inst_path.exists():
        return {p.id: 0 for p in manifest.properties}
    records = []
    with inst_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return count_property_coverage_from_records(records, manifest)


def template_distribution_from_jsonl(inst_path: Path) -> Counter:
    if not inst_path.exists():
        return Counter()
    records = []
    with inst_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return template_distribution_from_records(records)

"""Operator CLI: discover the CURIE inventory of a LibV2 course corpus.

Wraps ``lib.ontology.curie_discovery.discover_curies_from_corpus`` with
output formats targeted at the typical operator workflows:

  - ``--format table`` (default): human-readable, ranked by frequency.
    Use this to eyeball what a corpus actually contains.
  - ``--format json``: machine-readable ``{curie: count}`` dict, sorted.
    Pipe into jq, scripts, or other tooling.
  - ``--format manifest``: emit a property_manifest.<family>.yaml
    skeleton wired with sensible defaults. Operator hand-reviews labels
    + surface_forms before committing. Frequency-tier-aware: high-freq
    (>50) gets min_pairs=5, mid-freq (10-50) gets 3, low-freq (2-10)
    gets 2 — mirrors the rdf-shacl-551-2 manifest's calibrated tiers.

Use cases:

  1. Authoring a manifest for a new course family. Run with
     ``--format manifest > /tmp/draft.yaml``, hand-review, then move
     to ``schemas/training/property_manifest.<family>.yaml`` and
     commit.

  2. Catching manifest drift on an existing course. Run with
     ``--exclude-known-manifest`` to surface only the CURIEs the
     corpus uses that aren't yet declared.

  3. Driving the backfill loop dynamically. The
     ``backfill_form_data`` CLI accepts ``--discover-from-corpus``
     directly; this discovery CLI is for inspection / authoring, not
     a precondition of the loop.

Exit codes:

  0  ran cleanly, results emitted.
  1  chunks.jsonl not found / unreadable for the requested course.
  2  argument validation failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.curie_discovery import (  # noqa: E402
    diff_against_manifest,
    discover_curies_from_corpus,
)
from lib.ontology.property_manifest import (  # noqa: E402
    _family_slug,
    load_property_manifest,
)

logger = logging.getLogger(__name__)


def _resolve_chunks_jsonl(course_code: str) -> Optional[Path]:
    """Mirror of backfill_form_data._resolve_chunks_jsonl so this CLI
    doesn't have to import the larger backfill module.

    Phase 7c: prefers ``imscc_chunks/`` and falls back to legacy
    ``corpus/`` for unprovisioned archives.
    """
    candidates = [
        PROJECT_ROOT / "LibV2" / "courses" / course_code / "imscc_chunks" / "chunks.jsonl",
        PROJECT_ROOT / "LibV2" / "courses" / course_code / "corpus" / "chunks.jsonl",
        PROJECT_ROOT / "LibV2" / "courses" / course_code / "chunks.jsonl",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _tier_min_pairs(count: int) -> int:
    """Map a chunk-count to the manifest's tier-calibrated min_pairs.

    Mirrors the rdf-shacl-551-2 manifest's authored tier system so a
    discovery-emitted skeleton arrives wire-compatible with the
    structural validator's assumptions.
    """
    if count > 50:
        return 5
    if count >= 10:
        return 3
    return 2


def _format_as_table(
    discovered: Dict[str, int],
    *,
    title: str,
) -> str:
    if not discovered:
        return f"{title}\n  (no CURIEs above the frequency threshold)\n"
    lines: List[str] = [title, ""]
    width = max(len(c) for c in discovered) + 2
    lines.append(f"  {'CURIE'.ljust(width)} CHUNKS")
    lines.append(f"  {'-' * (width + 6)}")
    for curie, count in discovered.items():
        lines.append(f"  {curie.ljust(width)} {count}")
    return "\n".join(lines) + "\n"


def _format_as_manifest(
    discovered: Dict[str, int],
    *,
    family: str,
) -> str:
    """Emit a property_manifest.<family>.yaml skeleton.

    Only fields the validator strictly requires get populated. Optional
    fields (``learner_persona``, ``description``, ``min_accuracy``)
    are omitted so the operator's hand-review pass is the surface that
    decides them. Surface forms default to ``[curie]`` (single-form);
    the operator widens them as needed.
    """
    lines: List[str] = [
        f"family: {family}",
        f"description: Auto-generated skeleton from corpus discovery; hand-review before committing.",
        "properties:",
    ]
    for curie, count in discovered.items():
        prefix, _, local = curie.partition(":")
        prop_id = f"{prefix}_{local.lower()}"
        lines.extend([
            f"  - id: {prop_id}",
            f"    uri: \"\"  # TODO: paste canonical URI",
            f"    curie: {curie}",
            f"    label: {curie}",
            f"    surface_forms: [{curie!r}]",
            f"    min_pairs: {_tier_min_pairs(count)}",
            f"    # corpus_chunk_count: {count}",
        ])
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discover_curies",
        description=(
            "Walk a course's chunks.jsonl and report every CURIE the "
            "corpus contains, ranked by chunk-occurrence frequency. "
            "Emits human tables, machine JSON, or property_manifest "
            "skeletons depending on --format."
        ),
    )
    parser.add_argument(
        "--course-code",
        required=True,
        help="LibV2 course slug (e.g. 'rdf-shacl-551-2').",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help=(
            "Drop CURIEs appearing in fewer chunks than this. Default "
            "2 — matches the manifest's lowest tier."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "manifest"),
        default="table",
        help="Output shape. Default 'table' (human-readable).",
    )
    parser.add_argument(
        "--output",
        default="-",
        help=(
            "Output file path; '-' (default) writes to stdout. Use a "
            "path when piping into a manifest authoring step."
        ),
    )
    parser.add_argument(
        "--exclude-known-manifest",
        action="store_true",
        help=(
            "When set, omit CURIEs already declared in the family's "
            "property manifest. Surfaces only the gap — the new "
            "vocabulary the corpus uses that the manifest hasn't "
            "captured yet. Falls back to no-filter when the manifest "
            "doesn't exist."
        ),
    )
    parser.add_argument(
        "--text-fields",
        default="text",
        help=(
            "Comma-separated list of chunk-object fields to scan for "
            "CURIEs. Default 'text'. Pass 'text,title,description' to "
            "widen the scan surface."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.min_frequency < 1:
        print(
            f"ERROR: --min-frequency must be >= 1; got {args.min_frequency}",
            file=sys.stderr,
        )
        return 2

    chunks_path = _resolve_chunks_jsonl(args.course_code)
    if chunks_path is None:
        print(
            f"ERROR: chunks.jsonl not found for course "
            f"{args.course_code!r}; expected at "
            f"LibV2/courses/{args.course_code}/corpus/chunks.jsonl",
            file=sys.stderr,
        )
        return 1

    text_fields = tuple(
        f.strip() for f in args.text_fields.split(",") if f.strip()
    )
    if not text_fields:
        print(
            "ERROR: --text-fields produced an empty list",
            file=sys.stderr,
        )
        return 2

    try:
        discovered = discover_curies_from_corpus(
            chunks_path,
            min_frequency=args.min_frequency,
            text_fields=text_fields,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    family = _family_slug(args.course_code)

    if args.exclude_known_manifest:
        try:
            manifest = load_property_manifest(args.course_code)
            manifest_curies = [p.curie for p in manifest.properties]
            new, dropped = diff_against_manifest(discovered, manifest_curies)
            discovered = new
            if dropped:
                print(
                    f"# Note: {len(dropped)} manifest CURIE(s) absent "
                    f"from corpus: {dropped}",
                    file=sys.stderr,
                )
        except FileNotFoundError:
            print(
                f"# Note: no manifest at "
                f"property_manifest.{family}.yaml; "
                f"--exclude-known-manifest is a no-op.",
                file=sys.stderr,
            )

    if args.format == "table":
        title = (
            f"# Corpus CURIE inventory for {args.course_code} "
            f"(family={family}, min_frequency={args.min_frequency}, "
            f"distinct_curies={len(discovered)})"
        )
        rendered = _format_as_table(discovered, title=title)
    elif args.format == "json":
        rendered = json.dumps(discovered, indent=2) + "\n"
    elif args.format == "manifest":
        rendered = _format_as_manifest(discovered, family=family)
    else:  # pragma: no cover
        raise ValueError(f"unreachable: format={args.format!r}")

    if args.output == "-":
        sys.stdout.write(rendered)
        sys.stdout.flush()
    else:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(
            f"Wrote {len(discovered)} CURIE(s) ({args.format}) to "
            f"{args.output}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

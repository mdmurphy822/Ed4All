#!/usr/bin/env python3
"""
validate_page_objectives.py

Assert that each generated Courseforge HTML page's ``learningObjectives``
JSON-LD block references only IDs that are declared for that page's week
by the canonical objectives registry.

This guards against the LO-fanout defect where every week's pages emitted
the same invented week-local IDs that later collapsed onto the same four
canonical IDs in Trainforge's chunker.

Usage:
    python validate_page_objectives.py \
        --objectives inputs/exam-objectives/SAMPLE_101_objectives.json \
        --pages exports/SAMPLE_101_COURSE/03_content_development

    # Validate a single page:
    python validate_page_objectives.py \
        --objectives inputs/exam-objectives/SAMPLE_101_objectives.json \
        --pages exports/SAMPLE_101_COURSE/03_content_development/week_03/week_03_overview.html

Exit code 0 on success, 1 if any page violates the invariant.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the canonical resolver so the two emit/validate sides never drift.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from generate_course import load_canonical_objectives, resolve_week_objectives

JSON_LD_RE = re.compile(
    r'<script\s+type="application/ld\+json"\s*>\s*(\{.*?\})\s*</script>',
    re.DOTALL | re.IGNORECASE,
)
WEEK_PATH_RE = re.compile(r"week[_-]?(\d{1,2})", re.IGNORECASE)


def extract_json_ld_blocks(html: str) -> List[Dict[str, Any]]:
    """Return all parsed JSON-LD blocks from the given HTML source."""
    blocks: List[Dict[str, Any]] = []
    for match in JSON_LD_RE.finditer(html):
        raw = match.group(1)
        try:
            blocks.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return blocks


def extract_lo_ids(html: str) -> Optional[List[str]]:
    """Return the list of learningObjectives IDs from a page's JSON-LD, if any.

    Returns ``None`` if the page has no JSON-LD block or no ``learningObjectives``
    field. Returns ``[]`` if the field is present but empty. Case is preserved;
    callers normalize if they want case-insensitive comparison.
    """
    for block in extract_json_ld_blocks(html):
        los = block.get("learningObjectives")
        if los is None:
            continue
        ids: List[str] = []
        for lo in los:
            lo_id = lo.get("id") if isinstance(lo, dict) else None
            if lo_id:
                ids.append(str(lo_id))
        return ids
    return None


def infer_week_from_path(page_path: Path) -> Optional[int]:
    """Infer the 1-based week number from a page path like ``week_07/...``."""
    for part in reversed(page_path.parts):
        m = WEEK_PATH_RE.search(part)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def validate_page(
    page_path: Path,
    canonical: Dict[str, Any],
    week_num: Optional[int] = None,
) -> Tuple[bool, str]:
    """Validate one HTML page's JSON-LD ``learningObjectives``.

    Returns ``(ok, message)``. On success ``message`` is a short summary; on
    failure it names the offending IDs.
    """
    html = page_path.read_text(encoding="utf-8")
    ids = extract_lo_ids(html)
    if ids is None:
        # Pages without JSON-LD (e.g. non-content pages) are treated as pass.
        return True, f"{page_path.name}: no learningObjectives JSON-LD (skipped)"

    if week_num is None:
        week_num = infer_week_from_path(page_path)
    if week_num is None:
        return False, (
            f"{page_path}: could not infer week number from path; pass --week N"
        )

    allowed = {o["id"] for o in resolve_week_objectives(week_num, canonical)}
    if not allowed:
        # No canonical LOs declared for this week at all; flag so callers know.
        return True, (
            f"{page_path.name}: week {week_num} has no canonical LOs declared; "
            f"emitted {len(ids)} id(s) accepted without restriction"
        )

    ids_set = set(ids)
    extraneous = ids_set - allowed
    if extraneous:
        return False, (
            f"{page_path}: week {week_num} LO JSON-LD references "
            f"{sorted(extraneous)} which are NOT declared for this week. "
            f"Allowed for week {week_num}: {sorted(allowed)}"
        )

    return True, f"{page_path.name}: ok ({len(ids_set)} LO id(s), week {week_num})"


def discover_html_pages(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.html") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that Courseforge page learningObjectives JSON-LD blocks "
            "reference only IDs declared for that page's week in the canonical "
            "objectives JSON."
        )
    )
    parser.add_argument(
        "--objectives",
        required=True,
        help="Path to the canonical objectives JSON.",
    )
    parser.add_argument(
        "--pages",
        required=True,
        help="Path to a single HTML page or a directory tree of generated pages.",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Force a specific week number (useful when validating a single page).",
    )
    args = parser.parse_args()

    canonical = load_canonical_objectives(Path(args.objectives))
    root = Path(args.pages)
    pages = discover_html_pages(root)
    if not pages:
        print(f"No HTML pages found under {root}", file=sys.stderr)
        return 1

    failures: List[str] = []
    for page in pages:
        ok, msg = validate_page(page, canonical, week_num=args.week)
        print(("  OK  " if ok else "FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    print()
    print(f"Checked {len(pages)} page(s); {len(failures)} failure(s).")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

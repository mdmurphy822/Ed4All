#!/usr/bin/env python3
"""
Package multi-file weekly course content into an IMS Common Cartridge (IMSCC) file.

Walks 03_content_development/week_*/ directories and creates an IMSCC with a proper
imsmanifest.xml reflecting the week -> module hierarchy.

Per-week ``learningObjectives`` validation runs by default (Wave 2, Worker L
— REC-CTR-03). Every ``week_*/*.html`` page with JSON-LD is validated against
the canonical objectives registry before packaging; the packager refuses to
build when any page's ``learningObjectives`` lists an ID outside its week's
allowed set. This guards against the LO-fanout defect that shipped in
pre-Worker-H packages and capped Trainforge quality metrics.

Resolution order for the objectives file:

    1. Explicit ``--objectives PATH`` argument.
    2. Auto-discovery: ``<content_dir>/course.json`` if it exists.
    3. None available — log a warning and skip validation (backward-compat).

``--skip-validation`` remains as an explicit opt-out for emergencies.

Usage:
    python package_multifile_imscc.py <content_dir> <output_imscc>
    python package_multifile_imscc.py <content_dir> <output_imscc> \
        --objectives inputs/exam-objectives/SAMPLE_101_objectives.json
    python package_multifile_imscc.py <content_dir> <output_imscc> \
        --skip-validation  # escape hatch, not recommended for production
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# Match the ``<h1>Week N Overview: {real title}</h1>`` tag emitted by
# :func:`Courseforge.scripts.generate_course.generate_week`. The real
# chapter title — the part after ``"Overview:"`` / ``"Overview &mdash;"`` /
# ``"— Overview"`` — is what the manifest week item should surface so
# Brightspace / Canvas render a meaningful week label instead of a bare
# ``"Week 3"``.
_WEEK_OVERVIEW_H1_RE = re.compile(
    r"<h1[^>]*>\s*(.*?)\s*</h1>",
    re.IGNORECASE | re.DOTALL,
)
_OVERVIEW_TITLE_SEP_RE = re.compile(
    r"(?i)(?:overview\s*[:—–-]\s*|"      # "Overview: Title" / "Overview — Title"
    r"\s*[—–-]\s*overview\s*$)"          # "Title — Overview"
)
_BARE_OVERVIEW_RE = re.compile(r"(?i)^\s*overview\s*$")


def _extract_week_title(week_dir: Path, week_num: int) -> str:
    """Derive a human-readable week title from the week's overview HTML.

    Looks at ``week_NN_overview.html`` and pulls the chapter-title portion
    out of the emitted ``<h1>`` (Courseforge generate_week wraps it as
    ``"Week {N} Overview: {title}"``). Returns ``"Week {N}"`` when:

      * the overview file is missing,
      * its ``<h1>`` has no chapter title (neutral "Overview" fallback), or
      * parsing fails for any I/O reason.

    Never raises — packager manifest building is best-effort on the title
    layer; the LO-contract validator is the real gate for package quality.
    """
    overview_path = week_dir / f"week_{week_num:02d}_overview.html"
    if not overview_path.exists():
        return f"Week {week_num}"
    try:
        html = overview_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return f"Week {week_num}"
    m = _WEEK_OVERVIEW_H1_RE.search(html)
    if not m:
        return f"Week {week_num}"
    raw = m.group(1).strip()
    # Strip HTML entities that commonly appear in the H1 ("&mdash;").
    raw = raw.replace("&mdash;", "—").replace("&ndash;", "–")
    # Strip inner tags the H1 might carry (span wrappers, etc.).
    raw = re.sub(r"<[^>]+>", "", raw).strip()

    # Split off the "Week N Overview" prefix/suffix to isolate the real title.
    # Try "Week N Overview: Title" first.
    m2 = re.match(
        rf"(?i)^week\s+{week_num}\s*(?:overview)?\s*[:—–-]\s*(.+)$",
        raw,
    )
    if m2:
        title = m2.group(1).strip()
    else:
        m3 = re.match(
            rf"(?i)^(.+?)\s*[—–-]\s*week\s+{week_num}\s*(?:overview)?\s*$",
            raw,
        )
        if m3:
            title = m3.group(1).strip()
        else:
            title = raw

    # Bare "Overview" / empty → neutral week label (content-gen emits this
    # when no topic binds to the week).
    if not title or _BARE_OVERVIEW_RE.match(title):
        return f"Week {week_num}"
    return f"Week {week_num}: {title}"


def build_manifest(content_dir: Path, course_code: str, course_title: str) -> str:
    """Build imsmanifest.xml for multi-file weekly content."""
    ns = "http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
    lom_ns = "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource"
    lom_manifest_ns = "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest"

    # Register namespaces for clean serialization
    ET.register_namespace("", ns)
    ET.register_namespace("lom", lom_ns)
    ET.register_namespace("lomimscc", lom_manifest_ns)

    # Helper to create elements in the default IMSCC namespace
    def cc(tag):
        return f"{{{ns}}}{tag}"

    def lm(tag):
        return f"{{{lom_manifest_ns}}}{tag}"

    manifest = ET.Element(cc("manifest"), {
        "identifier": f"{course_code}_manifest",
    })

    # Metadata
    metadata = ET.SubElement(manifest, cc("metadata"))
    ET.SubElement(metadata, cc("schema")).text = "IMS Common Cartridge"
    ET.SubElement(metadata, cc("schemaversion")).text = "1.3.0"
    lom_el = ET.SubElement(metadata, lm("lom"))
    general = ET.SubElement(lom_el, lm("general"))
    title_el = ET.SubElement(general, lm("title"))
    ET.SubElement(title_el, lm("string"), {"language": "en"}).text = f"{course_code}: {course_title}"
    desc_el = ET.SubElement(general, lm("description"))
    ET.SubElement(desc_el, lm("string"), {"language": "en"}).text = (
        "A 12-week graduate course covering learning theory, instructional design, "
        "cognitive load, blended teaching, assessment, and accessibility."
    )

    # Organizations
    organizations = ET.SubElement(manifest, cc("organizations"))
    org = ET.SubElement(organizations, cc("organization"), {
        "identifier": "ORG_1",
        "structure": "rooted-hierarchy",
    })
    root_item = ET.SubElement(org, cc("item"), {"identifier": "ROOT"})
    ET.SubElement(root_item, cc("title")).text = f"{course_code}: {course_title}"

    # Resources
    resources = ET.SubElement(manifest, cc("resources"))

    # Walk week directories in order
    week_dirs = sorted(content_dir.glob("week_*"))
    for week_dir in week_dirs:
        if not week_dir.is_dir():
            continue
        week_name = week_dir.name
        week_num = week_name.replace("week_", "").lstrip("0") or "0"
        week_id = f"WEEK_{week_num}"

        week_item = ET.SubElement(root_item, cc("item"), {"identifier": week_id})
        # Prefer the real chapter title captured by generate_week in the
        # overview H1 (e.g. "Week 1: Fundamental Change in Education")
        # over the bare "Week N" label that earlier revisions emitted and
        # that produced an uninformative LMS week list.
        ET.SubElement(week_item, cc("title")).text = _extract_week_title(
            week_dir, int(week_num)
        )

        # Sort files: overview first, then content, application, self_check, summary, discussion
        order = {"overview": 0, "content": 1, "application": 2, "self_check": 3, "summary": 4, "discussion": 5}

        def sort_key(f):
            name = f.stem
            for key, val in order.items():  # noqa: B023
                if key in name:
                    return (val, name)
            return (99, name)

        html_files = sorted(week_dir.glob("*.html"), key=sort_key)

        for html_file in html_files:
            rel_path = f"{week_name}/{html_file.name}"
            res_id = re.sub(r"[^a-zA-Z0-9_]", "_", f"RES_{week_name}_{html_file.stem}")

            file_item = ET.SubElement(week_item, cc("item"), {
                "identifier": f"ITEM_{res_id}",
                "identifierref": res_id,
            })
            title_text = html_file.stem.replace(f"{week_name}_", "").replace("_", " ").title()
            ET.SubElement(file_item, cc("title")).text = title_text

            resource = ET.SubElement(resources, cc("resource"), {
                "identifier": res_id,
                "type": "webcontent",
                "href": rel_path,
            })
            ET.SubElement(resource, cc("file"), {"href": rel_path})

    ET.indent(manifest, space="  ")
    return ET.tostring(manifest, encoding="unicode", xml_declaration=True)


def validate_content_objectives(
    content_dir: Path, objectives_path: Path
) -> Tuple[bool, List[str]]:
    """Run `validate_page_objectives.validate_page` on every week_*/*.html page.

    Returns ``(ok, failure_messages)``. On success the failure list is empty.
    Pages without a JSON-LD block are passed over silently (validator's own
    rule). Imported lazily so packaging without --objectives incurs no cost.
    """
    from validate_page_objectives import (
        discover_html_pages,
        load_canonical_objectives,
        validate_page,
    )

    canonical = load_canonical_objectives(objectives_path)
    pages = discover_html_pages(content_dir)
    failures: List[str] = []
    for page in pages:
        # Only validate week_* pages; project docs and non-week HTML aren't
        # expected to carry LO metadata.
        if not any(part.startswith("week_") for part in page.parts):
            continue
        ok, msg = validate_page(page, canonical)
        if not ok:
            failures.append(msg)
    return (not failures, failures)


def package_imscc(
    content_dir: Path,
    output_path: Path,
    course_code: str,
    course_title: str,
    *,
    objectives_path: Optional[Path] = None,
    skip_validation: bool = False,
):
    """Create the IMSCC zip package.

    Per-week learningObjectives validation runs by default (Wave 2, Worker L
    — REC-CTR-03). Resolution order for the objectives file:

    1. Explicit ``objectives_path`` argument (CLI ``--objectives PATH``).
    2. Auto-discovery: ``content_dir / "course.json"`` if it exists.
    3. None available → log a warning and skip validation (backward-compat
       for callers that never wired the flag).

    ``skip_validation=True`` (CLI ``--skip-validation``) is an explicit
    opt-out that bypasses validation even when an objectives file is
    available. Hard-fail (``SystemExit(2)``) only occurs on a genuine
    validation FAILURE — never on a missing objectives file alone.
    """
    # Auto-discover objectives if not explicitly provided (default-on behavior).
    if objectives_path is None and not skip_validation:
        candidate = content_dir / "course.json"
        if candidate.exists():
            objectives_path = candidate
            print(f"[validate] Auto-discovered objectives at {candidate}")

    if skip_validation:
        print("[validate] SKIPPED (per --skip-validation) — build will not be gated on LO correctness.")
    elif objectives_path is None:
        print(
            "[validate] WARNING: no objectives file found; skipping LO validation. "
            "Pass --objectives or place course.json at content root to enable."
        )
    else:
        print(f"[validate] Checking per-week learningObjectives against {objectives_path.name}...")
        ok, failures = validate_content_objectives(content_dir, objectives_path)
        if not ok:
            print(f"[validate] REFUSING TO PACKAGE — {len(failures)} page(s) violate per-week LO contract:")
            for msg in failures:
                print(f"  - {msg}")
            print("Fix the offending pages (or re-run generate_course.py with --objectives) then retry.")
            print("Override with --skip-validation if you really know what you're doing.")
            raise SystemExit(2)
        print("[validate] All week pages pass per-week LO contract.")

    manifest_xml = build_manifest(content_dir, course_code, course_title)

    stub_included = False

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest_xml)

        # REC-TAX-01 cleanup (Wave 3, Worker M): bundle Worker J's
        # course_metadata.json classification stub at the zip root when
        # present. Trainforge consume already supports both zip-root and
        # sibling paths, but zip-root is the canonical self-contained
        # delivery — this closes the Wave 2 integration gap. Additive
        # only; absence is a no-op for backward-compat.
        stub_path = content_dir / "course_metadata.json"
        if stub_path.exists():
            zf.write(stub_path, stub_path.name)
            stub_included = True

        file_count = 0
        for week_dir in sorted(content_dir.glob("week_*")):
            if not week_dir.is_dir():
                continue
            for html_file in sorted(week_dir.glob("*.html")):
                zf.write(html_file, f"{week_dir.name}/{html_file.name}")
                file_count += 1

    print(f"IMSCC created: {output_path}")
    if stub_included:
        total = file_count + 2
        print(
            f"  Files: {file_count} HTML + 1 manifest + 1 course_metadata.json "
            f"= {total} total"
        )
    else:
        total = file_count + 1
        print(f"  Files: {file_count} HTML + 1 manifest = {total} total")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("content_dir", type=Path, help="Course content dir containing week_* subdirs")
    p.add_argument("output_imscc", type=Path, help="Output .imscc file path")
    p.add_argument("course_code", nargs="?", default="SAMPLE_101", help="Course code (default: SAMPLE_101)")
    p.add_argument("course_title", nargs="?", default="Sample Course",
                   help="Course title (default: Sample Course)")
    p.add_argument("--objectives", type=Path, default=None,
                   help=("Canonical objectives JSON to validate per-week LO "
                         "specificity before packaging. If omitted, auto-"
                         "discovered at <content_dir>/course.json when present."))
    p.add_argument("--skip-validation", action="store_true",
                   help=("Opt out of per-week LO validation (not recommended "
                         "for production builds)."))
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.output_imscc.parent.mkdir(parents=True, exist_ok=True)
    package_imscc(
        args.content_dir, args.output_imscc,
        args.course_code, args.course_title,
        objectives_path=args.objectives,
        skip_validation=args.skip_validation,
    )

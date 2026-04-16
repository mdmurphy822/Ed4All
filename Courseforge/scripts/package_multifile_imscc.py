#!/usr/bin/env python3
"""
Package multi-file weekly course content into an IMS Common Cartridge (IMSCC) file.

Walks 03_content_development/week_*/ directories and creates an IMSCC with a proper
imsmanifest.xml reflecting the week -> module hierarchy.

Usage:
    python package_multifile_imscc.py <content_dir> <output_imscc>
"""

import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


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
        ET.SubElement(week_item, cc("title")).text = f"Week {int(week_num)}"

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


def package_imscc(content_dir: Path, output_path: Path, course_code: str, course_title: str):
    """Create the IMSCC zip package."""
    manifest_xml = build_manifest(content_dir, course_code, course_title)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest_xml)

        file_count = 0
        for week_dir in sorted(content_dir.glob("week_*")):
            if not week_dir.is_dir():
                continue
            for html_file in sorted(week_dir.glob("*.html")):
                zf.write(html_file, f"{week_dir.name}/{html_file.name}")
                file_count += 1

    print(f"IMSCC created: {output_path}")
    print(f"  Files: {file_count} HTML + 1 manifest = {file_count + 1} total")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python package_multifile_imscc.py <content_dir> <output_imscc> [course_code] [course_title]")
        sys.exit(1)

    content_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    course_code = sys.argv[3] if len(sys.argv) > 3 else "DIGPED_101"
    course_title = sys.argv[4] if len(sys.argv) > 4 else "Foundations of Digital Pedagogy"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    package_imscc(content_dir, output_path, course_code, course_title)

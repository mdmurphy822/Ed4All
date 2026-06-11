#!/usr/bin/env python3
"""Deterministic renderer for the course-level Learning Objectives Map page.

Renders ``learning_objectives.html`` — a self-contained, zero-JS, WCAG 2.2 AA
accessible HTML page that surfaces the *canonical* synthesized learning
objectives of a course as first-class course content. The page lands in the
IMSCC under a new "Course Overview" module (see
``package_multifile_imscc.build_manifest``), so a learner — or a retrieval /
Q&A system asking "list the objectives this course teaches" — gets the real
``TO-NN`` / ``CO-NN`` structure instead of a scraped-together answer.

This is a **deterministic** transformation (no LLM). It consumes the
``synthesized_objectives.json`` emitted by the ``course_planning`` phase
(``{project}/01_learning_objectives/synthesized_objectives.json``) and honestly
renders whatever objective shape is present:

  * **Courseforge form** — ``terminal_objectives[] {id, statement, ...}`` +
    ``chapter_objectives`` as a dict keyed by chapter (``{"1": [CO, ...]}``) or
    as a flat list. Each CO may carry ``terminal_id`` parent linkage.
  * **LibV2 archive form** — ``terminal_outcomes[]`` + ``component_objectives[]``
    with a ``parent_terminal`` back-pointer. Normalized to the Courseforge form
    on read.

When a CO carries no resolvable parent terminal objective, it is rendered under
an explicit **"Unmapped objectives"** grouping rather than being silently
dropped or fabricated under an arbitrary TO — honesty over tidiness.

House style (matches sibling generated pages + Courseforge/CLAUDE.md
§ Metadata Output):

  * Bootstrap 4.3.1 CDN stylesheet link (no JS bundle).
  * ``<script type="application/ld+json">`` block with ``@context``
    ``https://ed4all.dev/ns/courseforge/v1`` carrying ``learningObjectives``.
  * ``data-cf-*`` pedagogy metadata on section / list-item wrappers
    (``data-cf-content-type``, ``data-cf-objective-id``, ``data-cf-bloom-level``,
    ``data-cf-cognitive-domain``).
  * Proper heading hierarchy (single ``<h1>`` → ``<h2>`` sections →
    ``<h3>`` per terminal objective), semantic ``<ul>`` / ``<table>``.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Canonical LO id patterns (mirror lib/ontology/learning_objectives.py).
_TO_RE = re.compile(r"^TO-\d{2,}$", re.IGNORECASE)
_CO_RE = re.compile(r"^CO-\d{2,}$", re.IGNORECASE)

_JSONLD_CONTEXT = "https://ed4all.dev/ns/courseforge/v1"

# Sentinel grouping key for component objectives with no resolvable parent.
_UNMAPPED = "__unmapped__"


def _esc(value: Any) -> str:
    """HTML-escape a value for safe interpolation into element text/attrs."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _norm_objectives(doc: Dict[str, Any]) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]]
]:
    """Normalize either objective shape to ``(terminals, components)`` lists.

    Each terminal dict is ``{id, statement, bloom_level?, cognitive_domain?,
    chapter?}``. Each component dict is ``{id, statement, bloom_level?,
    cognitive_domain?, terminal_id?, chapter?, section?, key_concepts?}``.

    Accepts:
      * Courseforge form: ``terminal_objectives`` + ``chapter_objectives``
        (dict-keyed-by-chapter OR flat list).
      * LibV2 archive form: ``terminal_outcomes`` + ``component_objectives``
        (flat list, ``parent_terminal`` back-pointer).
    """
    terminals_raw = (
        doc.get("terminal_objectives")
        or doc.get("terminal_outcomes")
        or []
    )
    terminals: List[Dict[str, Any]] = [
        t for t in terminals_raw if isinstance(t, dict)
    ]

    components: List[Dict[str, Any]] = []
    chapter_obj = doc.get("chapter_objectives")
    if isinstance(chapter_obj, dict):
        # Dict keyed by chapter -> stamp the chapter label onto each CO.
        for chapter_key in chapter_obj:
            bucket = chapter_obj[chapter_key]
            if not isinstance(bucket, list):
                continue
            for co in bucket:
                if not isinstance(co, dict):
                    continue
                co = dict(co)
                co.setdefault("chapter", chapter_key)
                components.append(co)
    elif isinstance(chapter_obj, list):
        components = [c for c in chapter_obj if isinstance(c, dict)]
    else:
        # LibV2 archive form.
        archive = doc.get("component_objectives")
        if isinstance(archive, list):
            for co in archive:
                if not isinstance(co, dict):
                    continue
                co = dict(co)
                # parent_terminal -> terminal_id (Courseforge field name).
                if "terminal_id" not in co and co.get("parent_terminal"):
                    co["terminal_id"] = co["parent_terminal"]
                components.append(co)

    return terminals, components


def _co_parent(co: Dict[str, Any]) -> Optional[str]:
    """Resolve a component objective's parent terminal id, or ``None``."""
    pid = co.get("terminal_id") or co.get("parent_terminal")
    if isinstance(pid, str) and _TO_RE.match(pid.strip()):
        return pid.strip().upper()
    return None


def _co_key_concepts(co: Dict[str, Any]) -> List[str]:
    """Extract key-concept tags from a component objective (best-effort)."""
    kc = co.get("key_concepts") or co.get("keyConcepts") or []
    if isinstance(kc, str):
        kc = [t.strip() for t in re.split(r"[;,]", kc) if t.strip()]
    return [str(t).strip() for t in kc if str(t).strip()]


def _chapter_label(co: Dict[str, Any]) -> str:
    """Human-readable chapter/section label for a component objective."""
    chapter = co.get("chapter")
    section = co.get("section")
    if section:
        return f"Section {_esc(section)}"
    if chapter is not None and str(chapter).strip():
        return f"Chapter {_esc(chapter)}"
    return ""


def _group_components(
    components: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group component objectives by parent terminal id.

    Components with no resolvable parent land under the ``_UNMAPPED`` key.
    Preserves source order within each group.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for co in components:
        parent = _co_parent(co) or _UNMAPPED
        groups.setdefault(parent, []).append(co)
    return groups


def _render_jsonld(
    course_code: str,
    terminals: List[Dict[str, Any]],
    components: List[Dict[str, Any]],
) -> str:
    """Build the JSON-LD metadata block (canonical course-page shape)."""
    learning_objectives: List[Dict[str, Any]] = []
    for t in terminals:
        tid = str(t.get("id", "")).strip()
        if not tid:
            continue
        entry: Dict[str, Any] = {"id": tid}
        if t.get("statement"):
            entry["statement"] = t["statement"]
        if t.get("bloom_level"):
            entry["bloomLevel"] = t["bloom_level"]
        if t.get("cognitive_domain"):
            entry["cognitiveDomain"] = t["cognitive_domain"]
        learning_objectives.append(entry)
    for co in components:
        cid = str(co.get("id", "")).strip()
        if not cid:
            continue
        entry = {"id": cid}
        if co.get("statement"):
            entry["statement"] = co["statement"]
        if co.get("bloom_level"):
            entry["bloomLevel"] = co["bloom_level"]
        if co.get("cognitive_domain"):
            entry["cognitiveDomain"] = co["cognitive_domain"]
        parent = _co_parent(co)
        if parent:
            entry["partOf"] = parent
        learning_objectives.append(entry)

    payload = {
        "@context": _JSONLD_CONTEXT,
        "@type": "CoursePage",
        "courseCode": course_code,
        "weekNumber": 0,
        "moduleType": "overview",
        "pageId": "course_overview_learning_objectives",
        "learningObjectives": learning_objectives,
    }
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    return (
        '  <script type="application/ld+json">\n'
        + body
        + "\n  </script>"
    )


def _render_co_list_item(co: Dict[str, Any]) -> str:
    """Render one component objective as a semantic ``<li>``."""
    cid = _esc(co.get("id", ""))
    statement = _esc(co.get("statement", ""))
    bloom = str(co.get("bloom_level", "")).strip().lower()
    domain = str(co.get("cognitive_domain", "")).strip().lower()
    chapter_label = _chapter_label(co)
    key_concepts = _co_key_concepts(co)

    attrs = ['class="list-group-item"']
    if cid:
        attrs.append(f'data-cf-objective-id="{cid}"')
    if bloom:
        attrs.append(f'data-cf-bloom-level="{_esc(bloom)}"')
    if domain:
        attrs.append(f'data-cf-cognitive-domain="{_esc(domain)}"')

    parts: List[str] = [f'        <li {" ".join(attrs)}>']
    parts.append(f"          <strong>{cid}</strong> &mdash; {statement}")
    meta_bits: List[str] = []
    if bloom:
        meta_bits.append(
            f'<span class="badge badge-secondary">Bloom: {_esc(bloom)}</span>'
        )
    if chapter_label:
        meta_bits.append(
            f'<span class="text-muted small">{chapter_label}</span>'
        )
    if meta_bits:
        parts.append(
            '          <div class="mt-1">' + " ".join(meta_bits) + "</div>"
        )
    if key_concepts:
        tags = " ".join(
            f'<span class="badge badge-light border">{_esc(t)}</span>'
            for t in key_concepts
        )
        parts.append(
            '          <div class="mt-1" aria-label="Key concepts">'
            + tags
            + "</div>"
        )
    parts.append("        </li>")
    return "\n".join(parts)


def _render_terminal_section(
    terminal: Dict[str, Any],
    children: List[Dict[str, Any]],
) -> str:
    """Render one terminal objective + its nested component objectives."""
    tid = _esc(terminal.get("id", ""))
    statement = _esc(terminal.get("statement", ""))
    bloom = str(terminal.get("bloom_level", "")).strip().lower()

    attrs = [
        'class="mb-4"',
        'data-cf-content-type="objectives"',
        'data-cf-teaching-role="introduce"',
    ]
    if tid:
        attrs.append(f'data-cf-objective-id="{tid}"')
    if bloom:
        attrs.append(f'data-cf-bloom-level="{_esc(bloom)}"')

    parts: List[str] = [f'      <div {" ".join(attrs)}>']
    parts.append(f"        <h3>{tid} &mdash; {statement}</h3>")
    if children:
        parts.append(
            "        <p>Supporting chapter objectives "
            f"({len(children)}):</p>"
        )
        parts.append('        <ul class="list-group list-group-flush">')
        for co in children:
            parts.append(_render_co_list_item(co))
        parts.append("        </ul>")
    else:
        parts.append(
            '        <p class="text-muted">'
            "No chapter objectives are mapped to this terminal objective."
            "</p>"
        )
    parts.append("      </div>")
    return "\n".join(parts)


def _render_summary_table(
    terminals: List[Dict[str, Any]],
    components: List[Dict[str, Any]],
) -> str:
    """Render the CO -> chapter -> TO summary mapping table."""
    rows: List[str] = []
    for co in components:
        cid = _esc(co.get("id", ""))
        parent = _co_parent(co)
        parent_cell = _esc(parent) if parent else (
            '<span class="text-muted">unmapped</span>'
        )
        chapter_label = _chapter_label(co) or (
            '<span class="text-muted">&mdash;</span>'
        )
        rows.append(
            "          <tr>\n"
            f"            <th scope=\"row\">{cid}</th>\n"
            f"            <td>{chapter_label}</td>\n"
            f"            <td>{parent_cell}</td>\n"
            "          </tr>"
        )
    if not rows:
        return ""
    return (
        '      <div class="mb-4" data-cf-content-type="summary">\n'
        "        <h2>Objective Map Summary</h2>\n"
        "        <p>Each chapter objective (CO) and the terminal objective "
        "(TO) it supports.</p>\n"
        '        <table class="table table-sm table-bordered">\n'
        "          <caption>Mapping of chapter objectives to chapters and "
        "terminal objectives.</caption>\n"
        "          <thead>\n"
        "            <tr>\n"
        '              <th scope="col">Chapter Objective</th>\n'
        '              <th scope="col">Chapter / Section</th>\n'
        '              <th scope="col">Terminal Objective</th>\n'
        "            </tr>\n"
        "          </thead>\n"
        "          <tbody>\n"
        + "\n".join(rows)
        + "\n          </tbody>\n"
        "        </table>\n"
        "      </div>"
    )


def render_learning_objectives_html(
    objectives: Dict[str, Any],
    *,
    course_code: str,
    course_title: str = "",
) -> str:
    """Render the full ``learning_objectives.html`` page string.

    Deterministic — same input always yields the same output. ``objectives``
    is the parsed ``synthesized_objectives.json`` doc (either supported shape).
    """
    title = (
        course_title
        or objectives.get("course_title")
        or objectives.get("course_name")
        or course_code
    )
    terminals, components = _norm_objectives(objectives)
    groups = _group_components(components)

    # Terminal sections in declared TO order.
    terminal_sections: List[str] = []
    for terminal in terminals:
        tid = str(terminal.get("id", "")).strip().upper()
        children = groups.get(tid, [])
        terminal_sections.append(_render_terminal_section(terminal, children))

    # Honest "Unmapped objectives" grouping for orphan component objectives.
    unmapped = groups.get(_UNMAPPED, [])
    if unmapped:
        attrs = (
            'class="mb-4" data-cf-content-type="objectives" '
            'data-cf-teaching-role="introduce"'
        )
        block = [f"      <div {attrs}>"]
        block.append("        <h3>Unmapped objectives</h3>")
        block.append(
            "        <p>These chapter objectives are not linked to a "
            "terminal objective in the course plan and are listed here "
            "for completeness.</p>"
        )
        block.append('        <ul class="list-group list-group-flush">')
        for co in unmapped:
            block.append(_render_co_list_item(co))
        block.append("        </ul>")
        block.append("      </div>")
        terminal_sections.append("\n".join(block))

    jsonld = _render_jsonld(course_code, terminals, components)
    summary_table = _render_summary_table(terminals, components)

    n_to = len(terminals)
    n_co = len(components)

    head = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, '
        'initial-scale=1.0">\n'
        f"  <title>Learning Objectives Map &mdash; {_esc(title)}</title>\n"
        '  <link rel="stylesheet" '
        'href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/'
        'bootstrap.min.css">\n'
        f"{jsonld}\n"
        "</head>\n"
    )

    body_parts: List[str] = [
        "<body>",
        '  <main class="container mt-4 mb-5">',
        '    <header class="mb-4">',
        f"      <h1>Learning Objectives Map</h1>",
        f'      <p class="lead text-muted">{_esc(title)}</p>',
        "      <p>This page lists every learning objective this course "
        f"teaches: {n_to} terminal objective(s) (TO) and {n_co} chapter "
        "objective(s) (CO), with each chapter objective grouped under the "
        "terminal objective it supports.</p>",
        "    </header>",
    ]

    if terminal_sections:
        body_parts.append(
            '    <section aria-labelledby="objectives-heading" '
            'data-cf-content-type="objectives">'
        )
        body_parts.append(
            '      <h2 id="objectives-heading">Terminal Objectives and '
            "Supporting Chapter Objectives</h2>"
        )
        body_parts.extend(terminal_sections)
        body_parts.append("    </section>")
    else:
        body_parts.append(
            '    <p class="text-muted">No learning objectives are defined '
            "for this course.</p>"
        )

    if summary_table:
        body_parts.append(
            '    <section aria-labelledby="summary-heading">'
        )
        # Promote the summary table's inner h2 anchor.
        body_parts.append(summary_table)
        body_parts.append("    </section>")

    body_parts.append("  </main>")
    body_parts.append("</body>")
    body_parts.append("</html>")

    return head + "\n".join(body_parts) + "\n"


def load_objectives(path: Path) -> Dict[str, Any]:
    """Load and parse a ``synthesized_objectives.json`` document."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_learning_objectives_page(
    objectives_path: Path,
    output_path: Path,
    *,
    course_code: str,
    course_title: str = "",
) -> Path:
    """Render + write the page from an objectives JSON file. Returns the path."""
    objectives = load_objectives(objectives_path)
    htmltext = render_learning_objectives_html(
        objectives, course_code=course_code, course_title=course_title
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(htmltext, encoding="utf-8")
    return output_path


if __name__ == "__main__":  # pragma: no cover - thin CLI
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("objectives_json", type=Path)
    p.add_argument("output_html", type=Path)
    p.add_argument("--course-code", default="SAMPLE_101")
    p.add_argument("--course-title", default="")
    args = p.parse_args()
    out = write_learning_objectives_page(
        args.objectives_json,
        args.output_html,
        course_code=args.course_code,
        course_title=args.course_title,
    )
    print(f"Wrote {out}")

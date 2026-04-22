#!/usr/bin/env python3
"""
Courseforge Course Generator

Generates multi-file weekly course modules from structured content data
and Courseforge HTML templates. Each week produces:
  - overview.html (objectives, readings, estimated time)
  - content_XX_topic.html (one per major concept, 600+ words each)
  - application.html (activities, worked examples)
  - self_check.html (interactive quiz with JS feedback)
  - summary.html (key takeaways, reflection questions)
  - discussion.html (forum prompt with guidelines)

Usage:
    python generate_course.py SAMPLE_101_course_data.json output_dir/
"""

import argparse
import html as html_mod
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is importable so lib.ontology.bloom resolves when
# this script is invoked from inside Courseforge/scripts/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.ontology.bloom import get_verbs_list as _get_canonical_verbs_list  # noqa: E402
from lib.ontology.slugs import canonical_slug as _slugify  # noqa: E402
from lib.ontology.taxonomy import validate_classification  # noqa: E402
from lib.ontology.teaching_roles import map_role as _map_teaching_role  # noqa: E402


# ---------------------------------------------------------------------------
# Canonical-objectives loading & per-week LO resolution
# ---------------------------------------------------------------------------
#
# The content-generation input (``<course>_course_data.json``) historically
# invented week-local objective IDs like ``W07-CO-01`` on each week's page,
# and those IDs were independently numbered ``01..04`` per week. Trainforge
# strips the ``W0N-`` prefix when normalizing ``learning_outcome_refs``, so
# every week's chunks collapsed onto the same four canonical IDs
# (``CO-01..CO-04``) and 24 of 28 declared outcomes ended up uncovered.
#
# The canonical source of truth is the ``inputs/exam-objectives/`` JSON
# (Terminal Objectives ``TO-*`` plus per-chapter Chapter Objectives
# ``CO-*`` grouped by ``chapter`` strings like ``"Week 3-4: Visual Design"``).
# When a caller passes ``--objectives <path>`` to ``generate_course.py`` we
# replace each week's objectives with the canonical subset for that week:
# every Terminal Objective plus every Chapter Objective whose chapter-range
# covers the week. The emitted JSON-LD then references globally-unique,
# canonical IDs that Trainforge can resolve against ``course.json``.

_WEEK_RANGE_RE = re.compile(r"[Ww]eek\s+(\d+)(?:\s*-\s*(\d+))?")


def _co_to_generator_format(co: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a canonical objective dict (from the objectives JSON) to the
    shape this script's renderers consume.

    The objectives JSON uses ``bloomLevel`` (camelCase); the renderer expects
    ``bloom_level`` (snake_case). ``key_concepts`` is left untouched.
    """
    out: Dict[str, Any] = {
        "id": co["id"],
        "statement": co["statement"],
    }
    bloom = co.get("bloomLevel") or co.get("bloom_level")
    if bloom:
        out["bloom_level"] = bloom
    verb = co.get("bloomVerb") or co.get("bloom_verb")
    if verb:
        out["bloom_verb"] = verb
    key_concepts = co.get("keyConcepts") or co.get("key_concepts")
    if key_concepts:
        out["key_concepts"] = key_concepts
    prereqs = co.get("prerequisiteObjectives") or co.get("prerequisite_objectives")
    if prereqs:
        out["prerequisite_objectives"] = prereqs
    return out


def load_canonical_objectives(objectives_path: Path) -> Dict[str, Any]:
    """Load the canonical objectives JSON (e.g. ``SAMPLE_101_objectives.json``)
    and return a structure keyed for per-week LO resolution.

    Returns a dict with keys:
        ``terminal_objectives``: list of TO dicts in generator format.
        ``week_to_chapter_objectives``: ``{int week_num: [CO dicts]}``.

    Chapter mapping uses the same regex Trainforge uses in
    ``Trainforge.process_course.load_objectives`` so the two stay in sync.
    """
    with open(objectives_path) as f:
        data = json.load(f)

    terminal = [_co_to_generator_format(o) for o in data.get("terminal_objectives", [])]

    week_to_cos: Dict[int, List[Dict[str, Any]]] = {}
    for chapter in data.get("chapter_objectives", []):
        chapter_name = chapter.get("chapter", "")
        m = _WEEK_RANGE_RE.search(chapter_name)
        if not m:
            continue
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        cos = [_co_to_generator_format(o) for o in chapter.get("objectives", [])]
        for w in range(start, end + 1):
            week_to_cos.setdefault(w, []).extend(cos)

    return {
        "terminal_objectives": terminal,
        "week_to_chapter_objectives": week_to_cos,
    }


def resolve_week_objectives(
    week_num: int, canonical: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Return the canonical LO list (TOs plus week-specific COs) for a week.

    If no chapter objectives cover ``week_num`` (e.g. course-overview pages
    keyed at ``week_num=0`` or a gap between declared chapters) we return
    only the terminal objectives, which always apply across the course.
    """
    terminal = canonical.get("terminal_objectives", []) or []
    chapter_cos = canonical.get("week_to_chapter_objectives", {}).get(week_num, []) or []
    # Preserve order (TOs first, then COs) and deduplicate by ID in case a
    # CO appears in more than one chapter range.
    seen: set = set()
    result: List[Dict[str, Any]] = []
    for o in list(terminal) + list(chapter_cos):
        if o["id"] in seen:
            continue
        seen.add(o["id"])
        result.append(o)
    return result

# ---------------------------------------------------------------------------
# Bloom's taxonomy detection
# ---------------------------------------------------------------------------

# Source of truth: schemas/taxonomies/bloom_verbs.json (loaded via
# lib.ontology.bloom). Migrated in Wave 1.2 / Worker H (REC-BL-01).
BLOOM_VERBS: Dict[str, List[str]] = _get_canonical_verbs_list()

# Cognitive domain inference from content type / Bloom's level
BLOOM_TO_DOMAIN: Dict[str, str] = {
    "remember": "factual",
    "understand": "conceptual",
    "apply": "procedural",
    "analyze": "conceptual",
    "evaluate": "metacognitive",
    "create": "procedural",
}


def detect_bloom_level(objective_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Detect Bloom's taxonomy level and verb from objective text.

    Returns (bloom_level, bloom_verb) or (None, None) if not detected.
    """
    text_lower = objective_text.lower().strip()
    for level, verbs in BLOOM_VERBS.items():
        for verb in verbs:
            if text_lower.startswith(verb) or f" {verb} " in text_lower:
                return level, verb
    return None, None


# `_slugify` is imported at the top of the module from
# ``lib.ontology.slugs.canonical_slug`` per REC-ID-03 (Wave 4, Worker Q). The
# alias preserves the local callers' ``_slugify(...)`` spelling.


# ---------------------------------------------------------------------------
# Courseforge CSS (matches user-edited Week 1 style)
COURSEFORGE_CSS = """
    body { font-family: system-ui, -apple-system, sans-serif; line-height: 1.7; max-width: 52em; margin: 0 auto; padding: 1.5em; color: #1a1a1a; }
    .skip-link { position: absolute; left: -9999px; } .skip-link:focus { position: static; }
    h1 { font-size: 1.8em; color: #1a365d; border-bottom: 3px solid #2c5aa0; padding-bottom: 0.3em; }
    h2 { font-size: 1.4em; color: #2c5aa0; margin-top: 1.8em; }
    h3 { font-size: 1.15em; color: #2d3748; margin-top: 1.3em; }
    .objectives { background: #ebf8ff; border-left: 4px solid #2c5aa0; padding: 1em 1.5em; margin: 1.5em 0; border-radius: 0 4px 4px 0; }
    .objectives h2 { color: #2c5aa0; margin-top: 0; }
    .key-term { font-weight: 700; color: #2d3748; }
    .callout { background: #f7fafc; border: 1px solid #e2e8f0; padding: 1em 1.5em; margin: 1em 0; border-radius: 4px; }
    .callout-warning { background: #fffbeb; border-color: #ffc107; }
    .callout-success { background: #f0fff4; border-color: #28a745; }
    .reflection { background: #fefcbf; border-left: 4px solid #d69e2e; padding: 1em 1.5em; margin: 1.5em 0; border-radius: 0 4px 4px 0; }
    .activity-card { background: #f8f9fa; border: 2px solid #2c5aa0; border-radius: 8px; padding: 1.5em; margin: 1em 0; }
    .activity-card h3 { color: #2c5aa0; margin-top: 0; }
    table { border-collapse: collapse; width: 100%; margin: 1em 0; }
    th { background: #2c5aa0; color: white; padding: 0.6em 1em; text-align: left; }
    td { padding: 0.6em 1em; border-bottom: 1px solid #e0e0e0; }
    tr:nth-child(even) { background: #f8f9fa; }
    .flip-card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1em; margin: 1.5em 0; }
    .flip-card { perspective: 600px; height: 180px; cursor: pointer; }
    .flip-card-inner { position: relative; width: 100%; height: 100%; transition: transform 0.6s; transform-style: preserve-3d; }
    .flip-card.flipped .flip-card-inner { transform: rotateY(180deg); }
    .flip-card-front, .flip-card-back { position: absolute; width: 100%; height: 100%; backface-visibility: hidden; border-radius: 8px; padding: 1em; display: flex; align-items: center; justify-content: center; text-align: center; box-sizing: border-box; }
    .flip-card-front { background: #2c5aa0; color: white; font-weight: 700; font-size: 1.1em; }
    .flip-card-back { background: #ebf8ff; color: #1a365d; transform: rotateY(180deg); font-size: 0.95em; border: 2px solid #2c5aa0; }
    .self-check { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1.5em; margin: 1.5em 0; }
    .self-check h3 { margin-top: 0; }
    .sc-option { display: block; padding: 0.5em; margin: 0.3em 0; border-radius: 4px; cursor: pointer; }
    .sc-option:hover { background: #ebf8ff; }
    .sc-option.correct { background: #d4edda; border: 1px solid #28a745; }
    .sc-option.incorrect { background: #f8d7da; border: 1px solid #dc3545; }
    .sc-feedback { display: none; padding: 0.5em; margin-top: 0.5em; border-radius: 4px; font-style: italic; }
    .discussion-prompt { background: #e8f4f8; border: 2px solid #2c5aa0; border-radius: 8px; padding: 1.5em; margin: 1em 0; }
    @media (prefers-color-scheme: dark) {
      body { background: #1a202c; color: #e2e8f0; }
      h1 { color: #90cdf4; border-color: #4299e1; }
      h2 { color: #90cdf4; }
      h3 { color: #cbd5e0; }
      .objectives { background: #2a4365; border-color: #4299e1; }
      .callout { background: #2d3748; border-color: #4a5568; }
      .reflection { background: #744210; border-color: #d69e2e; }
      .activity-card { background: #2d3748; border-color: #4299e1; }
      th { background: #2a4365; }
      td { border-color: #4a5568; }
      tr:nth-child(even) { background: #2d3748; }
      .flip-card-front { background: #2a4365; }
      .flip-card-back { background: #1a365d; color: #e2e8f0; border-color: #4299e1; }
      .self-check { background: #2d3748; border-color: #4a5568; }
      .discussion-prompt { background: #2a4365; border-color: #4299e1; }
    }
    @media (prefers-reduced-motion: reduce) {
      .flip-card-inner { transition: none; }
    }
"""

FLIP_CARD_JS = """
<script>
document.querySelectorAll('.flip-card').forEach(card => {
  card.addEventListener('click', () => card.classList.toggle('flipped'));
  card.addEventListener('keydown', e => { if(e.key==='Enter'||e.key===' '){e.preventDefault();card.classList.toggle('flipped');} });
});
</script>
"""

SELF_CHECK_JS = """
<script>
document.querySelectorAll('.self-check').forEach(sc => {
  const options = sc.querySelectorAll('.sc-option');
  const feedbacks = sc.querySelectorAll('.sc-feedback');
  let answered = false;
  options.forEach(opt => {
    opt.addEventListener('click', () => {
      if (answered) return;
      answered = true;
      const isCorrect = opt.dataset.correct === 'true';
      opt.classList.add(isCorrect ? 'correct' : 'incorrect');
      options.forEach(o => { if(o.dataset.correct==='true') o.classList.add('correct'); });
      feedbacks.forEach(f => f.style.display = 'block');
    });
  });
});
</script>
"""


def _wrap_page(title: str, course_code: str, week_num: int, body_html: str,
               extra_js: str = "",
               page_metadata: Optional[Dict[str, Any]] = None) -> str:
    """Wrap body content in a full HTML page with Courseforge styling.

    Args:
        page_metadata: Optional structured metadata dict rendered as JSON-LD
                       in <head> for downstream Trainforge extraction.
    """
    safe_title = html_mod.escape(title)
    json_ld = ""
    if page_metadata:
        json_ld = (
            '\n  <script type="application/ld+json">\n'
            + json.dumps(page_metadata, indent=2, ensure_ascii=False)
            + "\n  </script>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title} &mdash; {course_code}</title>
  <style>{COURSEFORGE_CSS}</style>{json_ld}
</head>
<body>
  <a href="#main-content" class="skip-link" data-cf-role="template-chrome">Skip to main content</a>
  <header role="banner" data-cf-role="template-chrome">
    <p>{course_code} &mdash; Week {week_num}</p>
  </header>
  <main id="main-content" role="main">
    <h1>{safe_title}</h1>
{body_html}
  </main>
  <footer role="contentinfo" data-cf-role="template-chrome">
    <p>&copy; 2026 {course_code}. All rights reserved.</p>
  </footer>
{extra_js}
</body>
</html>"""


def _source_attr_string(
    source_ids: Optional[List[str]],
    source_primary: Optional[str] = None,
) -> str:
    """Render the ``data-cf-source-ids`` / ``data-cf-source-primary`` attrs.

    Wave 9 source-provenance emit surface (P2 decision: section / heading /
    component wrappers only; no per-``<p>`` / ``<li>`` / ``<tr>`` bloat).
    Callers pass the sourceId list for the enclosing element; an empty or
    None list produces an empty string so the renderer stays identical to
    legacy behavior for non-textbook workflows.
    """
    if not source_ids:
        return ""
    joined = ",".join(html_mod.escape(sid) for sid in source_ids if sid)
    out = f' data-cf-source-ids="{joined}"'
    if source_primary:
        out += f' data-cf-source-primary="{html_mod.escape(source_primary)}"'
    return out


def _render_objectives(
    objectives: List[Dict],
    *,
    source_ids: Optional[List[str]] = None,
    source_primary: Optional[str] = None,
) -> str:
    """Render a learning objectives box with data-cf-* metadata attributes.

    Wave 9: when ``source_ids`` is non-empty, the enclosing
    ``.objectives`` wrapper carries ``data-cf-source-ids`` (and optionally
    ``data-cf-source-primary``) so downstream consumers can tie the block
    back to a DART source region.
    """
    items = []
    for o in objectives:
        bloom_level = o.get("bloom_level")
        bloom_verb = o.get("bloom_verb")
        if not bloom_level:
            bloom_level, bloom_verb = detect_bloom_level(o["statement"])
        domain = BLOOM_TO_DOMAIN.get(bloom_level, "conceptual") if bloom_level else ""
        attrs = f' data-cf-objective-id="{html_mod.escape(o["id"])}"'
        if bloom_level:
            attrs += f' data-cf-bloom-level="{bloom_level}"'
        if bloom_verb:
            attrs += f' data-cf-bloom-verb="{bloom_verb}"'
        if domain:
            attrs += f' data-cf-cognitive-domain="{domain}"'
        items.append(
            f'      <li{attrs}><strong>{o["id"]}:</strong> {html_mod.escape(o["statement"])}</li>'
        )
    items_html = "\n".join(items)
    wrapper_source_attrs = _source_attr_string(source_ids, source_primary)
    return f"""
    <div class="objectives" role="region" aria-label="Learning Objectives"{wrapper_source_attrs}>
      <h2>Learning Objectives</h2>
      <p>After completing this module, you will be able to:</p>
      <ul>
{items_html}
      </ul>
    </div>"""


def _render_flip_cards(terms: List[Dict]) -> str:
    """Render a grid of flip cards for key terms with data-cf-* metadata."""
    # REC-VOC-02: deterministic teaching_role from (component, purpose) pair.
    fc_role = _map_teaching_role("flip-card", "term-definition")
    fc_role_attr = f' data-cf-teaching-role="{fc_role}"' if fc_role else ""
    cards = []
    for _i, t in enumerate(terms):
        front = html_mod.escape(t["term"])
        back = html_mod.escape(t["definition"])
        term_slug = _slugify(t["term"])
        cards.append(f"""
      <div class="flip-card" tabindex="0" role="button" aria-label="Flip card: {front}"
           data-cf-component="flip-card" data-cf-purpose="term-definition"{fc_role_attr}
           data-cf-term="{term_slug}">
        <div class="flip-card-inner">
          <div class="flip-card-front">{front}</div>
          <div class="flip-card-back">{back}</div>
        </div>
      </div>""")
    return f'    <div class="flip-card-grid">{"".join(cards)}\n    </div>'


def _render_self_check(
    questions: List[Dict],
    *,
    source_ids: Optional[List[str]] = None,
    source_primary: Optional[str] = None,
) -> str:
    """Render self-check quiz questions with JS feedback and data-cf-* metadata.

    Wave 9: each ``.self-check`` wrapper carries ``data-cf-source-ids``
    derived from the per-question ``source_references`` when declared, or
    the page-level ``source_ids`` otherwise. Emit happens at the wrapper
    level only (P2 decision).
    """
    blocks = []
    for i, q in enumerate(questions, 1):
        opts = []
        for _j, opt in enumerate(q["options"]):
            correct = "true" if opt.get("correct") else "false"
            fb = html_mod.escape(opt.get("feedback", ""))
            opts.append(
                f'        <label class="sc-option" data-correct="{correct}">'
                f'<input type="radio" name="q{i}" style="margin-right:0.5em">'
                f'{html_mod.escape(opt["text"])}</label>\n'
                f'        <div class="sc-feedback">{fb}</div>'
            )
        options_html = "\n".join(opts)
        # Build data-cf-* attributes for the self-check
        bloom = q.get("bloom_level", "remember")
        obj_ref = q.get("objective_ref", "")
        # REC-VOC-02: deterministic teaching_role from (component, purpose) pair.
        sc_role = _map_teaching_role("self-check", "formative-assessment")
        sc_role_attr = f' data-cf-teaching-role="{sc_role}"' if sc_role else ""
        sc_attrs = (
            f' data-cf-component="self-check" data-cf-purpose="formative-assessment"'
            f'{sc_role_attr}'
            f' data-cf-bloom-level="{bloom}"'
        )
        if obj_ref:
            sc_attrs += f' data-cf-objective-ref="{html_mod.escape(obj_ref)}"'
        q_refs = q.get("source_references")
        if q_refs:
            q_ids = _refs_to_id_list(q_refs)
            q_primary = _refs_primary(q_refs)
        else:
            q_ids = source_ids
            q_primary = source_primary
        sc_attrs += _source_attr_string(q_ids, q_primary)
        blocks.append(f"""
    <div class="self-check"{sc_attrs}>
      <h3>Question {i}</h3>
      <p>{html_mod.escape(q["question"])}</p>
{options_html}
    </div>""")
    return "\n".join(blocks)


def _infer_content_type(section: Dict) -> str:
    """Infer a content type label for a section from its structure/heading."""
    heading = section.get("heading", "").lower()
    if section.get("flip_cards"):
        return "definition"
    if any(kw in heading for kw in ("example", "case study", "scenario")):
        return "example"
    if any(kw in heading for kw in ("procedure", "steps", "how to", "process")):
        return "procedure"
    if any(kw in heading for kw in ("compare", "contrast", "versus", "vs")):
        return "comparison"
    if any(kw in heading for kw in ("activity", "exercise", "practice")):
        return "exercise"
    if any(kw in heading for kw in ("overview", "introduction")):
        return "overview"
    if any(kw in heading for kw in ("summary", "recap", "takeaway")):
        return "summary"
    return "explanation"


def _render_content_sections(
    sections: List[Dict],
    *,
    source_ids: Optional[List[str]] = None,
    source_primary: Optional[str] = None,
) -> str:
    """Render content sections with h2/h3 headings, data-cf-* metadata, and paragraphs.

    Wave 9 source-provenance (P2 decision): when a per-section source
    override is declared on ``section["source_references"]`` (list of
    SourceReference dicts), that section's heading carries its own
    ``data-cf-source-ids``. Otherwise the page-level ``source_ids`` are
    used for every heading that doesn't override. Never emitted on
    per-``<p>`` / ``<li>`` / ``<tr>`` children.
    """
    parts = []
    for section in sections:
        heading = html_mod.escape(section["heading"])
        level = section.get("level", 2)
        tag = f"h{level}"

        # Build data-cf-* attributes for the heading
        content_type = section.get("content_type") or _infer_content_type(section)
        key_term_slugs = ",".join(
            _slugify(t["term"] if isinstance(t, dict) else t)
            for t in section.get("flip_cards", section.get("key_terms", []))
        )
        bloom_range = section.get("bloom_range", "")
        h_attrs = f' data-cf-content-type="{content_type}"'
        if key_term_slugs:
            h_attrs += f' data-cf-key-terms="{key_term_slugs}"'
        if bloom_range:
            h_attrs += f' data-cf-bloom-range="{bloom_range}"'

        # Wave 9: per-section source override takes precedence over
        # page-level ids. Falls back to page ids when the section doesn't
        # declare its own mapping.
        section_refs = section.get("source_references")
        if section_refs:
            section_ids = _refs_to_id_list(section_refs)
            section_primary = _refs_primary(section_refs)
        else:
            section_ids = source_ids
            section_primary = source_primary
        section_attrs = _source_attr_string(section_ids, section_primary)
        h_attrs += section_attrs

        # Wave 35: wrap every h2/h3 + paragraph group in a <section>
        # carrying the same data-cf-source-ids so
        # :class:`ContentGroundingValidator` can walk each <p>'s
        # ancestors to find the grounding attribute. Pre-Wave-35 the
        # attribute lived only on the <h2>, which is a sibling of the
        # <p> in the DOM — validator's ancestor walk missed it and
        # flagged every body paragraph as ungrounded. Section-wrapping
        # preserves the Wave 9 invariant (attributes live on section /
        # heading / component wrappers, never on raw <p>/<li>/<tr>).
        if section_attrs:
            parts.append(f"    <section{section_attrs}>")
        parts.append(f"    <{tag}{h_attrs}>{heading}</{tag}>")
        for para in section.get("paragraphs", []):
            # Apply key-term markup
            p = para
            for term in section.get("key_terms", []):
                escaped = html_mod.escape(term)
                p = re.sub(
                    rf"\b({re.escape(escaped)})\b",
                    r'<strong class="key-term">\1</strong>',
                    p, count=1, flags=re.IGNORECASE
                )
            parts.append(f"    <p>{p}</p>")
        # Render any flip cards in this section
        if section.get("flip_cards"):
            parts.append(_render_flip_cards(section["flip_cards"]))
        # Render any callout
        if section.get("callout"):
            c = section["callout"]
            cls = f'callout {c.get("type", "")}'.strip()
            callout_type = "application-note" if c.get("type") == "callout-warning" else "note"
            parts.append(
                f'    <div class="{cls}" role="region"'
                f' aria-label="{html_mod.escape(c.get("label", "Note"))}"'
                f' data-cf-content-type="{callout_type}">'
            )
            parts.append(f'      <h3>{html_mod.escape(c.get("heading", "Note"))}</h3>')
            for item in c.get("items", []):
                parts.append(f"      <p>{item}</p>")
            if c.get("list"):
                parts.append("      <ul>")
                for li in c["list"]:
                    parts.append(f"        <li>{li}</li>")
                parts.append("      </ul>")
            parts.append("    </div>")
        # Render any table
        if section.get("table"):
            t = section["table"]
            parts.append("    <table>")
            if t.get("headers"):
                parts.append("      <thead><tr>" +
                    "".join(f"<th>{h}</th>" for h in t["headers"]) +
                    "</tr></thead>")
            parts.append("      <tbody>")
            for row in t.get("rows", []):
                parts.append("        <tr>" +
                    "".join(f"<td>{cell}</td>" for cell in row) +
                    "</tr>")
            parts.append("      </tbody></table>")
        # Wave 35: close the grounding <section> wrapper we may have
        # opened above the heading. No-op when no source_refs were in
        # play — pre-Wave-35 sections had no wrapper and we keep that
        # back-compat for non-textbook workflows.
        if section_attrs:
            parts.append("    </section>")
    return "\n".join(parts)


def _render_activities(
    activities: List[Dict],
    *,
    source_ids: Optional[List[str]] = None,
    source_primary: Optional[str] = None,
) -> str:
    """Render activity cards with data-cf-* metadata.

    Wave 9: per-activity ``source_references`` override the page-level
    ``source_ids`` when present. Emit site is the ``.activity-card``
    wrapper only (P2 decision).
    """
    parts = []
    # REC-VOC-02: deterministic teaching_role from (component, purpose) pair.
    act_role = _map_teaching_role("activity", "practice")
    act_role_attr = f' data-cf-teaching-role="{act_role}"' if act_role else ""
    for i, act in enumerate(activities, 1):
        bloom = act.get("bloom_level", "apply")
        obj_ref = act.get("objective_ref", "")
        act_attrs = (
            f' data-cf-component="activity" data-cf-purpose="practice"'
            f'{act_role_attr}'
            f' data-cf-bloom-level="{bloom}"'
        )
        if obj_ref:
            act_attrs += f' data-cf-objective-ref="{html_mod.escape(obj_ref)}"'
        act_refs = act.get("source_references")
        if act_refs:
            act_ids = _refs_to_id_list(act_refs)
            act_primary = _refs_primary(act_refs)
        else:
            act_ids = source_ids
            act_primary = source_primary
        act_attrs += _source_attr_string(act_ids, act_primary)
        parts.append(f"""
    <div class="activity-card"{act_attrs}>
      <h3>Activity {i}: {html_mod.escape(act["title"])}</h3>
      <p>{act["description"]}</p>
    </div>""")
    return "\n".join(parts)


def _render_reflection(questions: List[str]) -> str:
    """Render reflection/discussion section."""
    items = "\n".join(f"        <li>{q}</li>" for q in questions)
    return f"""
    <div class="reflection" role="region" aria-label="Reflection and Discussion">
      <h2>Reflection &amp; Discussion</h2>
      <ol>
{items}
      </ol>
    </div>"""


def _build_objectives_metadata(objectives: List[Dict]) -> List[Dict[str, Any]]:
    """Build structured objective metadata for JSON-LD from week objectives."""
    result = []
    for o in objectives:
        bloom_level = o.get("bloom_level")
        bloom_verb = o.get("bloom_verb")
        if not bloom_level:
            bloom_level, bloom_verb = detect_bloom_level(o["statement"])
        domain = BLOOM_TO_DOMAIN.get(bloom_level, "conceptual") if bloom_level else "conceptual"
        key_concepts = [_slugify(c) for c in o.get("key_concepts", []) if _slugify(c)]
        entry: Dict[str, Any] = {
            "id": o["id"],
            "statement": o["statement"],
            "bloomLevel": bloom_level,
            "bloomVerb": bloom_verb,
            "cognitiveDomain": domain,
        }
        if key_concepts:
            entry["keyConcepts"] = key_concepts
        # Include assessment suggestions based on Bloom's level
        if bloom_level and bloom_level in BLOOM_VERBS:
            from_bloom = {
                "remember": ["multiple_choice", "true_false", "fill_in_blank"],
                "understand": ["multiple_choice", "short_answer", "fill_in_blank"],
                "apply": ["multiple_choice", "short_answer", "essay"],
                "analyze": ["multiple_choice", "essay", "short_answer"],
                "evaluate": ["essay", "multiple_choice", "short_answer"],
                "create": ["essay", "short_answer"],
            }
            entry["assessmentSuggestions"] = from_bloom.get(bloom_level, [])
        prereqs = o.get("prerequisite_objectives", [])
        if prereqs:
            entry["prerequisiteObjectives"] = prereqs
        result.append(entry)
    return result


def _collect_section_roles(section: Dict) -> List[str]:
    """Collect deterministic teachingRole values for components inside a section.

    Walks the section's flip_cards / self_check / activities children (if
    present) and maps each (component, purpose) pair via
    `lib.ontology.teaching_roles.map_role`. Returns a sorted list (stable
    for diff-friendly JSON-LD output). Empty list when no tagged
    components are found.

    Sections today commonly carry only ``flip_cards`` inline, but we
    defensively handle ``self_check`` and ``activities`` so future
    structural changes don't silently drop role coverage.
    """
    roles: Set[str] = set()
    if section.get("flip_cards"):
        r = _map_teaching_role("flip-card", "term-definition")
        if r:
            roles.add(r)
    for _q in section.get("self_check", []) or []:
        r = _map_teaching_role("self-check", "formative-assessment")
        if r:
            roles.add(r)
            break  # one entry is enough to cover the section
    for _a in section.get("activities", []) or []:
        r = _map_teaching_role("activity", "practice")
        if r:
            roles.add(r)
            break
    return sorted(roles)


def _build_sections_metadata(sections: List[Dict]) -> List[Dict[str, Any]]:
    """Build structured section metadata for JSON-LD.

    Wave 9 addition (source provenance): each section may carry a
    ``source_references`` key (list of
    :class:`schemas/knowledge/source_reference.schema.json` SourceReference
    objects). When present and non-empty, emitted as ``sourceReferences``
    on the section entry. Absent / empty → elided for backward compat.
    """
    result = []
    for section in sections:
        content_type = section.get("content_type") or _infer_content_type(section)
        entry: Dict[str, Any] = {
            "heading": section["heading"],
            "contentType": content_type,
        }
        # Key terms from flip_cards
        if section.get("flip_cards"):
            entry["keyTerms"] = [
                {"term": t["term"], "definition": t["definition"]}
                for t in section["flip_cards"]
            ]
        # REC-VOC-02: deterministic teaching_role array collected from
        # tagged components inside the section (stable, diff-friendly order).
        teaching_roles = _collect_section_roles(section)
        if teaching_roles:
            entry["teachingRole"] = teaching_roles
        bloom_range = section.get("bloom_range")
        if bloom_range:
            entry["bloomRange"] = [bloom_range] if isinstance(bloom_range, str) else bloom_range
        # Wave 9: section-level source attribution (override pattern).
        section_refs = section.get("source_references")
        if section_refs:
            entry["sourceReferences"] = list(section_refs)
        result.append(entry)
    return result


def _build_page_metadata(
    course_code: str, week_num: int, module_type: str, page_id: str,
    objectives: Optional[List[Dict]] = None,
    sections: Optional[List[Dict]] = None,
    misconceptions: Optional[List[Dict]] = None,
    suggested_assessments: Optional[List[str]] = None,
    classification: Optional[Dict] = None,
    prerequisite_pages: Optional[List[str]] = None,
    source_references: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the JSON-LD metadata dict for a single page.

    Wave 2 additions (REC-TAX-01 + REC-JSL-02):
      * ``classification``: when non-empty, the course-level taxonomy
        block is inherited on every page's JSON-LD (``classification``
        key). Validated upstream in :func:`generate_course`.
      * ``prerequisite_pages``: when non-empty, emits the
        ``prerequisitePages`` array matching
        ``schemas/knowledge/courseforge_jsonld_v1.schema.json`` §58-62.

    Wave 9 addition (source provenance):
      * ``source_references``: optional list of SourceReference dicts
        (per ``schemas/knowledge/source_reference.schema.json``). Emitted
        as top-level ``sourceReferences`` when non-empty. Mirrors the
        ``prerequisite_pages`` elision pattern — empty / None → key
        omitted so the page still validates against the schema for
        non-textbook workflows (course_generation).
    """
    meta: Dict[str, Any] = {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": course_code,
        "weekNumber": week_num,
        "moduleType": module_type,
        "pageId": page_id,
    }
    if objectives:
        meta["learningObjectives"] = _build_objectives_metadata(objectives)
    if sections:
        meta["sections"] = _build_sections_metadata(sections)
    if misconceptions:
        meta["misconceptions"] = misconceptions
    if suggested_assessments:
        meta["suggestedAssessmentTypes"] = suggested_assessments
    if classification:
        meta["classification"] = classification
    if prerequisite_pages:
        meta["prerequisitePages"] = list(prerequisite_pages)
    if source_references:
        meta["sourceReferences"] = list(source_references)
    return meta


def _refs_to_id_list(refs: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Return the sourceId list implied by a SourceReference array.

    Shape matches ``schemas/knowledge/source_reference.schema.json``:
    every entry has a ``sourceId`` key. Missing / malformed entries are
    skipped silently — emit-side validation is the source-refs gate's
    job, not the renderer's.
    """
    if not refs:
        return []
    ids: List[str] = []
    for ref in refs:
        if isinstance(ref, dict):
            sid = ref.get("sourceId")
            if isinstance(sid, str) and sid:
                ids.append(sid)
    return ids


def _refs_primary(refs: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Return the single dominant sourceId when one exists.

    A ref with ``role == "primary"`` wins; when multiple primaries exist
    (or none) we return None so callers can skip the
    ``data-cf-source-primary`` attribute. Keeps the attribute honest —
    only emitted when routing produced an unambiguous dominant source.
    """
    if not refs:
        return None
    primary_ids = [
        ref.get("sourceId")
        for ref in refs
        if isinstance(ref, dict)
        and ref.get("role") == "primary"
        and isinstance(ref.get("sourceId"), str)
        and ref.get("sourceId")
    ]
    if len(primary_ids) == 1:
        return primary_ids[0]
    return None


def _page_refs_for(
    source_module_map: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    week_num: int,
    page_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Look up the SourceReference list for a given (week, page) pair.

    ``source_module_map`` follows the Wave 9 shape emitted by the
    ``source-router`` agent::

        {
          "week_03": {
            "content_01": {
              "primary":      ["dart:slug#s5_p2"],
              "contributing": ["dart:slug#s4_p0"],
              "confidence":   0.85
            },
            ...
          }
        }

    The function normalizes the ``primary`` / ``contributing`` lists into
    a SourceReference array. When the page is absent or the map is empty,
    returns ``None`` so the renderer elides all source-ref output for
    that page (backward-compat path).
    """
    if not source_module_map:
        return None
    week_key = f"week_{week_num:02d}"
    week_entries = source_module_map.get(week_key)
    if not isinstance(week_entries, dict):
        return None
    # Map may key by either the full page_id or a short key; prefer exact
    # match on the emitted page_id, fall back to stripping the week prefix,
    # then (Wave 35) to the short form with the slug suffix dropped so
    # content_NN_the_skills_in_a_digital_age matches the router's
    # content_NN entry.
    entry = week_entries.get(page_id)
    if entry is None:
        short_key = page_id
        prefix = f"week_{week_num:02d}_"
        if short_key.startswith(prefix):
            short_key = short_key[len(prefix):]
        entry = week_entries.get(short_key)
        if entry is None:
            # Drop the slug suffix: content_NN_<slug> → content_NN.
            slug_match = re.match(
                r"^(content_\d{2}|overview|application|self_check|summary)(?:_.*)?$",
                short_key,
            )
            if slug_match:
                entry = week_entries.get(slug_match.group(1))
        if entry is None and short_key.startswith("content_"):
            # Wave 35: router only emits a single ``content_01`` entry
            # per week; content_02..10 share the same DART source
            # region. Fall back to content_01 so every generated
            # content page inherits the week's grounding.
            entry = week_entries.get("content_01")
    if not isinstance(entry, dict):
        return None

    refs: List[Dict[str, Any]] = []
    confidence = entry.get("confidence")
    for sid in entry.get("primary") or []:
        if isinstance(sid, str) and sid:
            ref: Dict[str, Any] = {"sourceId": sid, "role": "primary"}
            if isinstance(confidence, (int, float)):
                ref["confidence"] = float(confidence)
            refs.append(ref)
    for sid in entry.get("contributing") or []:
        if isinstance(sid, str) and sid:
            ref = {"sourceId": sid, "role": "contributing"}
            if isinstance(confidence, (int, float)):
                ref["confidence"] = float(confidence)
            refs.append(ref)
    return refs or None


def generate_week(
    week_data: Dict,
    output_dir: Path,
    course_code: str,
    canonical_objectives: Optional[Dict[str, Any]] = None,
    classification: Optional[Dict] = None,
    prerequisite_map: Optional[Dict[str, List[str]]] = None,
    source_module_map: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
):
    """Generate all files for a single week.

    When ``canonical_objectives`` is provided (from
    :func:`load_canonical_objectives`), the week's ``objectives`` are
    overridden with the canonical TOs plus the Chapter Objectives declared
    for that week. This ensures emitted ``learningObjectives`` JSON-LD
    references globally-unique canonical IDs (e.g. ``CO-05``) instead of
    invented week-local IDs (``W03-CO-01``) that all collapse to the same
    four IDs after Trainforge's week-prefix normalization.

    Wave 2 additions (REC-TAX-01 + REC-JSL-02):
      * ``classification`` — course-level taxonomy block inherited on every
        page's JSON-LD (no-op when ``None``).
      * ``prerequisite_map`` — optional ``{page_id: [prereq_page_id, ...]}``
        map; non-empty entries surface as ``prerequisitePages`` arrays on
        the matching page's JSON-LD.

    Wave 9 addition (source provenance):
      * ``source_module_map`` — optional ``{week_key: {page_id: entry}}``
        map produced by the ``source-router`` agent. Populates JSON-LD
        ``sourceReferences[]`` and ``data-cf-source-ids`` attributes on
        each page's HTML. Absent / empty → no source attribution emitted;
        backward-compat for non-textbook workflows.
    """
    week_num = week_data["week_number"]
    week_dir = output_dir / f"week_{week_num:02d}"
    week_dir.mkdir(parents=True, exist_ok=True)
    prereq_lookup = prerequisite_map or {}

    # Override week objectives with canonical, week-specific LOs when a
    # canonical objectives registry is supplied. Falls back to the week's
    # declared objectives for backward compatibility with older callers.
    if canonical_objectives is not None:
        resolved = resolve_week_objectives(week_num, canonical_objectives)
        if resolved:
            week_data = dict(week_data)
            week_data["objectives"] = resolved

    # Remove old monolithic module.html
    old = week_dir / "module.html"
    if old.exists():
        old.unlink()

    # Shared week-level misconceptions (optional in input data)
    week_misconceptions = week_data.get("misconceptions", [])

    # 1. Overview
    overview_page_id = f"week_{week_num:02d}_overview"
    overview_refs = _page_refs_for(source_module_map, week_num, overview_page_id)
    overview_ids = _refs_to_id_list(overview_refs)
    overview_primary = _refs_primary(overview_refs)
    overview_body = _render_objectives(
        week_data["objectives"], source_ids=overview_ids,
        source_primary=overview_primary,
    )
    # Wave 41: wrap the overview body (non-objectives region) in a
    # <section data-cf-source-ids="…"> so
    # :class:`ContentGroundingValidator`'s ancestor walk finds grounding
    # on every body <p>/<li>. The objectives <div> already carries its
    # own grounding via _render_objectives; the extra <p> text +
    # readings list would otherwise be DOM siblings of a sourceless
    # <main>. No wrapper emitted when overview_ids is empty (preserves
    # the test_no_map_no_emit back-compat contract).
    overview_body_attrs = _source_attr_string(overview_ids, overview_primary)
    if overview_body_attrs:
        overview_body += f"\n    <section{overview_body_attrs}>"
    if week_data.get("overview_text"):
        for p in week_data["overview_text"]:
            overview_body += f"\n    <p>{p}</p>"
    if week_data.get("readings"):
        overview_body += "\n    <h2>Readings &amp; Resources</h2>\n    <ul>"
        for r in week_data["readings"]:
            overview_body += f"\n      <li>{r}</li>"
        overview_body += "\n    </ul>"
    overview_body += f"\n    <p><strong>Estimated time:</strong> {week_data.get('estimated_hours', '3-4')} hours</p>"
    if overview_body_attrs:
        overview_body += "\n    </section>"

    overview_meta = _build_page_metadata(
        course_code, week_num, "overview",
        overview_page_id,
        objectives=week_data["objectives"],
        classification=classification,
        prerequisite_pages=prereq_lookup.get(overview_page_id),
        source_references=overview_refs,
    )
    overview_html = _wrap_page(
        f"Week {week_num} Overview: {week_data['title']}",
        course_code, week_num, overview_body,
        page_metadata=overview_meta,
    )
    (week_dir / f"week_{week_num:02d}_overview.html").write_text(overview_html, encoding="utf-8")

    # 2. Content modules
    for ci, content in enumerate(week_data.get("content_modules", []), 1):
        slug = re.sub(r"[^a-z0-9]+", "_", content["title"].lower()).strip("_")[:40]
        page_id = f"week_{week_num:02d}_content_{ci:02d}_{slug}"
        page_refs = _page_refs_for(source_module_map, week_num, page_id)
        page_ids_list = _refs_to_id_list(page_refs)
        page_primary = _refs_primary(page_refs)
        content_body = _render_content_sections(
            content["sections"],
            source_ids=page_ids_list,
            source_primary=page_primary,
        )
        extra_js = FLIP_CARD_JS if any(
            s.get("flip_cards") for s in content["sections"]
        ) else ""
        content_meta = _build_page_metadata(
            course_code, week_num, "content", page_id,
            objectives=week_data["objectives"],
            sections=content["sections"],
            misconceptions=content.get("misconceptions", week_misconceptions),
            classification=classification,
            prerequisite_pages=prereq_lookup.get(page_id),
            source_references=page_refs,
        )
        content_html = _wrap_page(
            f"Week {week_num}: {content['title']}",
            course_code, week_num, content_body, extra_js,
            page_metadata=content_meta,
        )
        filename = f"{page_id}.html"
        (week_dir / filename).write_text(content_html, encoding="utf-8")

    # 3. Application / Activities
    if week_data.get("activities"):
        app_page_id = f"week_{week_num:02d}_application"
        app_refs = _page_refs_for(source_module_map, week_num, app_page_id)
        app_ids = _refs_to_id_list(app_refs)
        app_primary = _refs_primary(app_refs)
        # Wave 41: wrap the Learning Activities heading + any intro
        # prose in a <section data-cf-source-ids="…"> so the ancestor
        # walk finds grounding on the <h2> (and any future intro <p>).
        # The .activity-card wrappers already carry per-card source-ids,
        # but the opening <h2> would otherwise be a direct <main> child
        # with no grounding ancestor.
        app_body_attrs = _source_attr_string(app_ids, app_primary)
        app_body = ""
        if app_body_attrs:
            app_body += f"\n    <section{app_body_attrs}>"
        app_body += "\n    <h2>Learning Activities</h2>"
        app_body += _render_activities(
            week_data["activities"], source_ids=app_ids, source_primary=app_primary,
        )
        if app_body_attrs:
            app_body += "\n    </section>"
        app_meta = _build_page_metadata(
            course_code, week_num, "application",
            app_page_id,
            objectives=week_data["objectives"],
            suggested_assessments=["short_answer", "essay"],
            classification=classification,
            prerequisite_pages=prereq_lookup.get(app_page_id),
            source_references=app_refs,
        )
        app_html = _wrap_page(
            f"Week {week_num}: Application &amp; Activities",
            course_code, week_num, app_body,
            page_metadata=app_meta,
        )
        (week_dir / f"week_{week_num:02d}_application.html").write_text(app_html, encoding="utf-8")

    # 4. Self-check
    if week_data.get("self_check_questions"):
        sc_page_id = f"week_{week_num:02d}_self_check"
        sc_refs = _page_refs_for(source_module_map, week_num, sc_page_id)
        sc_ids = _refs_to_id_list(sc_refs)
        sc_primary = _refs_primary(sc_refs)
        # Wave 41: wrap the Self-Check heading + intro <p> in a
        # <section data-cf-source-ids="…">. The .self-check item
        # wrappers already carry source-ids, but the "Select the best
        # answer…" intro paragraph would otherwise be a direct <main>
        # child and flagged as ungrounded by
        # :class:`ContentGroundingValidator`'s ancestor walk.
        sc_body_attrs = _source_attr_string(sc_ids, sc_primary)
        sc_body = ""
        if sc_body_attrs:
            sc_body += f"\n    <section{sc_body_attrs}>"
        sc_body += "\n    <h2>Self-Check: Test Your Understanding</h2>"
        sc_body += "\n    <p>Select the best answer for each question. You will receive immediate feedback.</p>"
        sc_body += _render_self_check(
            week_data["self_check_questions"],
            source_ids=sc_ids,
            source_primary=sc_primary,
        )
        if sc_body_attrs:
            sc_body += "\n    </section>"
        sc_meta = _build_page_metadata(
            course_code, week_num, "assessment",
            sc_page_id,
            objectives=week_data["objectives"],
            suggested_assessments=["multiple_choice", "true_false"],
            classification=classification,
            prerequisite_pages=prereq_lookup.get(sc_page_id),
            source_references=sc_refs,
        )
        sc_html = _wrap_page(
            f"Week {week_num}: Self-Check Quiz",
            course_code, week_num, sc_body, SELF_CHECK_JS,
            page_metadata=sc_meta,
        )
        (week_dir / f"week_{week_num:02d}_self_check.html").write_text(sc_html, encoding="utf-8")

    # 5. Summary
    summary_page_id = f"week_{week_num:02d}_summary"
    summary_refs = _page_refs_for(source_module_map, week_num, summary_page_id)
    summary_ids = _refs_to_id_list(summary_refs)
    summary_primary = _refs_primary(summary_refs)
    summary_heading_attrs = _source_attr_string(summary_ids, summary_primary)
    # Wave 41: wrap the entire summary body (key takeaways list,
    # reflection block, next-week preview) in a <section
    # data-cf-source-ids="…">. Pre-Wave-41 only the <h2> carried
    # grounding, leaving the <li> takeaways + preview <p> as direct
    # <main> children with no grounding ancestor. The Wave 9 pattern
    # (attributes on section / component wrappers only, never on raw
    # <p>/<li>/<tr>) is preserved. No wrapper when summary_ids is
    # empty (preserves the test_no_map_no_emit back-compat contract).
    summary_body = ""
    if summary_heading_attrs:
        summary_body += f"\n    <section{summary_heading_attrs}>"
    summary_body += f"\n    <h2{summary_heading_attrs}>Key Takeaways</h2>"
    if week_data.get("key_takeaways"):
        summary_body += "\n    <ul>"
        for kt in week_data["key_takeaways"]:
            summary_body += f"\n      <li>{kt}</li>"
        summary_body += "\n    </ul>"
    if week_data.get("reflection_questions"):
        summary_body += _render_reflection(week_data["reflection_questions"])
    if week_data.get("next_week_preview"):
        summary_body += f"\n    <h2>Looking Ahead</h2>\n    <p>{week_data['next_week_preview']}</p>"
    if summary_heading_attrs:
        summary_body += "\n    </section>"

    summary_meta = _build_page_metadata(
        course_code, week_num, "summary",
        summary_page_id,
        objectives=week_data["objectives"],
        classification=classification,
        prerequisite_pages=prereq_lookup.get(summary_page_id),
        source_references=summary_refs,
    )
    summary_html = _wrap_page(
        f"Week {week_num}: Summary &amp; Reflection",
        course_code, week_num, summary_body,
        page_metadata=summary_meta,
    )
    (week_dir / f"week_{week_num:02d}_summary.html").write_text(summary_html, encoding="utf-8")

    # 6. Discussion
    if week_data.get("discussion"):
        disc = week_data["discussion"]
        disc_page_id = f"week_{week_num:02d}_discussion"
        disc_refs = _page_refs_for(source_module_map, week_num, disc_page_id)
        disc_ids = _refs_to_id_list(disc_refs)
        disc_primary = _refs_primary(disc_refs)
        disc_attrs = _source_attr_string(disc_ids, disc_primary)
        disc_body = f"""
    <div class="discussion-prompt"{disc_attrs}>
      <h2>Discussion Forum</h2>
      <p>{disc["prompt"]}</p>
      <h3>Guidelines</h3>
      <ul>
        <li><strong>Initial Post:</strong> {disc.get("initial_post", "250 words minimum")}</li>
        <li><strong>Replies:</strong> {disc.get("replies", "Respond to at least 2 classmates (100 words each)")}</li>
        <li><strong>Due:</strong> {disc.get("due", "Initial post by Wednesday; replies by Sunday")}</li>
      </ul>
    </div>"""
        disc_meta = _build_page_metadata(
            course_code, week_num, "discussion",
            disc_page_id,
            objectives=week_data["objectives"],
            classification=classification,
            prerequisite_pages=prereq_lookup.get(disc_page_id),
            source_references=disc_refs,
        )
        disc_html = _wrap_page(
            f"Week {week_num}: Discussion",
            course_code, week_num, disc_body,
            page_metadata=disc_meta,
        )
        (week_dir / f"week_{week_num:02d}_discussion.html").write_text(disc_html, encoding="utf-8")

    # Count files generated
    files = list(week_dir.glob("*.html"))
    return len(files), [f.name for f in sorted(files)]


def generate_course(
    course_data_path: str,
    output_dir: str,
    objectives_path: Optional[str] = None,
    classification: Optional[Dict] = None,
    source_module_map_path: Optional[str] = None,
):
    """Generate a full course from a JSON data file.

    Args:
        course_data_path: Path to the course data JSON (per-week content,
            activities, self-checks, etc.).
        output_dir: Directory to write the generated ``week_XX/`` folders.
        objectives_path: Optional path to the canonical objectives JSON
            (e.g. ``Courseforge/inputs/exam-objectives/SAMPLE_101_objectives.json``).
            When provided, each page's ``learningObjectives`` JSON-LD is
            emitted using canonical CO / TO IDs resolved from the week
            mapping declared in the objectives JSON. Pass ``None`` to
            preserve the previous behaviour and use whatever ``objectives``
            list the course data JSON provides for each week.
        classification: Optional course-level subject-taxonomy block
            (Wave 2 REC-TAX-01). Overrides any ``classification`` key
            declared in the course data JSON. When non-empty, the block
            is validated against ``schemas/taxonomies/taxonomy.json``
            BEFORE any files are written — fail-closed. Non-empty
            classification triggers emission of:
              * ``course_metadata.json`` at ``output_dir`` root, and
              * a ``classification`` key on every page's JSON-LD.
        source_module_map_path: Optional path to a Wave 9
            ``source_module_map.json`` produced by the ``source-router``
            agent. Shape: ``{week_key: {page_id: {primary: [...],
            contributing: [...], confidence: 0.x}}}``. Populates
            ``sourceReferences[]`` in JSON-LD and ``data-cf-source-ids``
            on HTML wrappers. Absent / empty → no provenance emit
            (backward compat).
    """
    data = json.loads(Path(course_data_path).read_text())
    out = Path(output_dir)
    course_code = data.get("course_code", "COURSE_101")

    # Resolve effective classification: CLI/caller arg wins over course-data JSON.
    effective_classification = classification
    if effective_classification is None:
        effective_classification = data.get("classification") or None

    # Fail-closed validation: a non-empty classification block must match
    # the authoritative taxonomy before ANY file is written.
    if effective_classification:
        errors = validate_classification(effective_classification)
        if errors:
            raise ValueError(
                "Invalid classification for course "
                f"{course_code}: {'; '.join(errors)}"
            )

    # Optional prerequisite map: {page_id: [prereq_page_id, ...]}.
    # Sourced from course data; empty/missing → no prerequisitePages emitted.
    prerequisite_map = data.get("prerequisite_map") or {}

    # Wave 9: optional source-routing map. Prefer explicit CLI path, then
    # course-data override, then no map at all (backward-compat path).
    source_module_map: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    if source_module_map_path:
        map_path = Path(source_module_map_path)
        if map_path.exists():
            source_module_map = json.loads(map_path.read_text(encoding="utf-8"))
    elif isinstance(data.get("source_module_map"), dict):
        source_module_map = data.get("source_module_map")

    canonical = None
    if objectives_path:
        canonical = load_canonical_objectives(Path(objectives_path))
        tos = len(canonical.get("terminal_objectives", []))
        cos = sum(len(v) for v in canonical.get("week_to_chapter_objectives", {}).values())
        print(
            f"Loaded canonical objectives: {tos} terminal objective(s), "
            f"{cos} chapter-objective week-slot assignment(s) across weeks "
            f"{sorted(canonical['week_to_chapter_objectives'].keys())}"
        )

    total_files = 0
    for week in data["weeks"]:
        count, files = generate_week(
            week, out, course_code,
            canonical_objectives=canonical,
            classification=effective_classification,
            prerequisite_map=prerequisite_map,
            source_module_map=source_module_map,
        )
        total_files += count
        print(f"  Week {week['week_number']:2d}: {count} files - {', '.join(files)}")

    # Emit course-level classification stub (REC-TAX-01). Only emitted when
    # classification is populated — preserves backward compat for existing
    # pipelines that never declared a taxonomy.
    if effective_classification:
        out.mkdir(parents=True, exist_ok=True)
        stub = {
            "course_code": course_code,
            "course_title": data.get("course_title") or data.get("title") or course_code,
            "classification": {
                "division": effective_classification.get("division"),
                "primary_domain": effective_classification.get("primary_domain"),
                "subdomains": list(effective_classification.get("subdomains") or []),
                "topics": list(effective_classification.get("topics") or []),
            },
            "ontology_mappings": {
                "acm_ccs": list(
                    (data.get("ontology_mappings") or {}).get("acm_ccs") or []
                ),
                "lcsh": list(
                    (data.get("ontology_mappings") or {}).get("lcsh") or []
                ),
            },
        }
        stub_path = out / "course_metadata.json"
        stub_path.write_text(json.dumps(stub, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote course classification stub: {stub_path}")

    print(f"\nTotal: {total_files} files generated")
    return total_files


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Courseforge HTML pages from structured course data."
    )
    parser.add_argument("course_data", help="Path to <course>_course_data.json")
    parser.add_argument("output_dir", help="Output directory for generated week folders")
    parser.add_argument(
        "--objectives",
        default=None,
        help=(
            "Optional path to the canonical objectives JSON "
            "(e.g. inputs/exam-objectives/<COURSE>_objectives.json). When "
            "provided, emitted learningObjectives JSON-LD uses canonical CO/TO "
            "IDs resolved per-week rather than the week-local IDs that the "
            "course-data file may carry. This is the recommended mode; it "
            "fixes the defect where every week's pages reference the same "
            "four LOs after Trainforge's week-prefix normalization."
        ),
    )
    # ---------------------------------------------------------------- #
    # Wave 2 REC-TAX-01 classification flags. When both --division and
    # --primary-domain are provided, a course_metadata.json stub is
    # written at the output_dir root and a ``classification`` block is
    # inherited on every page's JSON-LD. The block is validated against
    # schemas/taxonomies/taxonomy.json (fail-closed). CLI flags override
    # any ``classification`` declared in the course-data JSON.
    # ---------------------------------------------------------------- #
    parser.add_argument(
        "--division",
        default=None,
        choices=["STEM", "ARTS"],
        help="Classification division (REC-TAX-01). Pair with --primary-domain.",
    )
    parser.add_argument(
        "--primary-domain",
        default=None,
        help=(
            "Classification primary domain slug (REC-TAX-01), e.g. "
            "computer-science. Required when --division is set."
        ),
    )
    parser.add_argument(
        "--subdomains",
        default="",
        help=(
            "Comma-separated subdomain slugs under the declared domain "
            "(REC-TAX-01), e.g. software-engineering,algorithms."
        ),
    )
    parser.add_argument(
        "--source-module-map",
        default=None,
        help=(
            "Wave 9: optional path to source_module_map.json produced by the "
            "source-router agent. Populates sourceReferences[] in JSON-LD and "
            "data-cf-source-ids on HTML wrappers. Absent → no provenance emit."
        ),
    )
    return parser


def _build_classification_from_args(args: argparse.Namespace) -> Optional[Dict]:
    """Assemble a classification dict from CLI flags, or None when absent.

    Returns ``None`` when neither ``--division`` nor ``--primary-domain`` is
    set, leaving the course-data JSON's ``classification`` (if any) as the
    source. Returns a populated dict when ``--division`` and ``--primary-domain``
    are both provided.
    """
    if not (args.division and args.primary_domain):
        return None
    subs = [s.strip() for s in (args.subdomains or "").split(",") if s.strip()]
    return {
        "division": args.division,
        "primary_domain": args.primary_domain,
        "subdomains": subs,
        "topics": [],
    }


if __name__ == "__main__":
    args = _build_cli_parser().parse_args()
    classification = _build_classification_from_args(args)
    generate_course(
        args.course_data,
        args.output_dir,
        objectives_path=args.objectives,
        classification=classification,
        source_module_map_path=args.source_module_map,
    )

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
    python generate_course.py DIGPED_101_course_data.json output_dir/
"""

import json
import html as html_mod
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


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
               extra_js: str = "") -> str:
    """Wrap body content in a full HTML page with Courseforge styling."""
    safe_title = html_mod.escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title} &mdash; {course_code}</title>
  <style>{COURSEFORGE_CSS}</style>
</head>
<body>
  <a href="#main-content" class="skip-link">Skip to main content</a>
  <header role="banner">
    <p>{course_code}: Foundations of Digital Pedagogy &mdash; Week {week_num}</p>
  </header>
  <main id="main-content" role="main">
    <h1>{safe_title}</h1>
{body_html}
  </main>
  <footer role="contentinfo">
    <p>&copy; 2026 {course_code}: Foundations of Digital Pedagogy. All rights reserved.</p>
  </footer>
{extra_js}
</body>
</html>"""


def _render_objectives(objectives: List[Dict]) -> str:
    """Render a learning objectives box."""
    items = "\n".join(
        f'      <li><strong>{o["id"]}:</strong> {html_mod.escape(o["statement"])}</li>'
        for o in objectives
    )
    return f"""
    <div class="objectives" role="region" aria-label="Learning Objectives">
      <h2>Learning Objectives</h2>
      <p>After completing this module, you will be able to:</p>
      <ul>
{items}
      </ul>
    </div>"""


def _render_flip_cards(terms: List[Dict]) -> str:
    """Render a grid of flip cards for key terms."""
    cards = []
    for i, t in enumerate(terms):
        front = html_mod.escape(t["term"])
        back = html_mod.escape(t["definition"])
        cards.append(f"""
      <div class="flip-card" tabindex="0" role="button" aria-label="Flip card: {front}">
        <div class="flip-card-inner">
          <div class="flip-card-front">{front}</div>
          <div class="flip-card-back">{back}</div>
        </div>
      </div>""")
    return f'    <div class="flip-card-grid">{"".join(cards)}\n    </div>'


def _render_self_check(questions: List[Dict]) -> str:
    """Render self-check quiz questions with JS feedback."""
    blocks = []
    for i, q in enumerate(questions, 1):
        opts = []
        for j, opt in enumerate(q["options"]):
            correct = "true" if opt.get("correct") else "false"
            fb = html_mod.escape(opt.get("feedback", ""))
            opts.append(
                f'        <label class="sc-option" data-correct="{correct}">'
                f'<input type="radio" name="q{i}" style="margin-right:0.5em">'
                f'{html_mod.escape(opt["text"])}</label>\n'
                f'        <div class="sc-feedback">{fb}</div>'
            )
        options_html = "\n".join(opts)
        blocks.append(f"""
    <div class="self-check">
      <h3>Question {i}</h3>
      <p>{html_mod.escape(q["question"])}</p>
{options_html}
    </div>""")
    return "\n".join(blocks)


def _render_content_sections(sections: List[Dict]) -> str:
    """Render content sections with h2/h3 headings and paragraphs."""
    parts = []
    for section in sections:
        heading = html_mod.escape(section["heading"])
        level = section.get("level", 2)
        tag = f"h{level}"
        parts.append(f"    <{tag}>{heading}</{tag}>")
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
            parts.append(f'    <div class="{cls}" role="region" aria-label="{html_mod.escape(c.get("label", "Note"))}">')
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
    return "\n".join(parts)


def _render_activities(activities: List[Dict]) -> str:
    """Render activity cards."""
    parts = []
    for i, act in enumerate(activities, 1):
        parts.append(f"""
    <div class="activity-card">
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


def generate_week(week_data: Dict, output_dir: Path, course_code: str):
    """Generate all files for a single week."""
    week_num = week_data["week_number"]
    week_dir = output_dir / f"week_{week_num:02d}"
    week_dir.mkdir(parents=True, exist_ok=True)

    # Remove old monolithic module.html
    old = week_dir / "module.html"
    if old.exists():
        old.unlink()

    # 1. Overview
    overview_body = _render_objectives(week_data["objectives"])
    if week_data.get("overview_text"):
        for p in week_data["overview_text"]:
            overview_body += f"\n    <p>{p}</p>"
    if week_data.get("readings"):
        overview_body += "\n    <h2>Readings &amp; Resources</h2>\n    <ul>"
        for r in week_data["readings"]:
            overview_body += f"\n      <li>{r}</li>"
        overview_body += "\n    </ul>"
    overview_body += f"\n    <p><strong>Estimated time:</strong> {week_data.get('estimated_hours', '3-4')} hours</p>"

    overview_html = _wrap_page(
        f"Week {week_num} Overview: {week_data['title']}",
        course_code, week_num, overview_body
    )
    (week_dir / f"week_{week_num:02d}_overview.html").write_text(overview_html, encoding="utf-8")

    # 2. Content modules
    for ci, content in enumerate(week_data.get("content_modules", []), 1):
        slug = re.sub(r"[^a-z0-9]+", "_", content["title"].lower()).strip("_")[:40]
        content_body = _render_content_sections(content["sections"])
        extra_js = FLIP_CARD_JS if any(
            s.get("flip_cards") for s in content["sections"]
        ) else ""
        content_html = _wrap_page(
            f"Week {week_num}: {content['title']}",
            course_code, week_num, content_body, extra_js
        )
        filename = f"week_{week_num:02d}_content_{ci:02d}_{slug}.html"
        (week_dir / filename).write_text(content_html, encoding="utf-8")

    # 3. Application / Activities
    if week_data.get("activities"):
        app_body = "\n    <h2>Learning Activities</h2>"
        app_body += _render_activities(week_data["activities"])
        app_html = _wrap_page(
            f"Week {week_num}: Application &amp; Activities",
            course_code, week_num, app_body
        )
        (week_dir / f"week_{week_num:02d}_application.html").write_text(app_html, encoding="utf-8")

    # 4. Self-check
    if week_data.get("self_check_questions"):
        sc_body = "\n    <h2>Self-Check: Test Your Understanding</h2>"
        sc_body += "\n    <p>Select the best answer for each question. You will receive immediate feedback.</p>"
        sc_body += _render_self_check(week_data["self_check_questions"])
        sc_html = _wrap_page(
            f"Week {week_num}: Self-Check Quiz",
            course_code, week_num, sc_body, SELF_CHECK_JS
        )
        (week_dir / f"week_{week_num:02d}_self_check.html").write_text(sc_html, encoding="utf-8")

    # 5. Summary
    summary_body = "\n    <h2>Key Takeaways</h2>"
    if week_data.get("key_takeaways"):
        summary_body += "\n    <ul>"
        for kt in week_data["key_takeaways"]:
            summary_body += f"\n      <li>{kt}</li>"
        summary_body += "\n    </ul>"
    if week_data.get("reflection_questions"):
        summary_body += _render_reflection(week_data["reflection_questions"])
    if week_data.get("next_week_preview"):
        summary_body += f"\n    <h2>Looking Ahead</h2>\n    <p>{week_data['next_week_preview']}</p>"

    summary_html = _wrap_page(
        f"Week {week_num}: Summary &amp; Reflection",
        course_code, week_num, summary_body
    )
    (week_dir / f"week_{week_num:02d}_summary.html").write_text(summary_html, encoding="utf-8")

    # 6. Discussion
    if week_data.get("discussion"):
        disc = week_data["discussion"]
        disc_body = f"""
    <div class="discussion-prompt">
      <h2>Discussion Forum</h2>
      <p>{disc["prompt"]}</p>
      <h3>Guidelines</h3>
      <ul>
        <li><strong>Initial Post:</strong> {disc.get("initial_post", "250 words minimum")}</li>
        <li><strong>Replies:</strong> {disc.get("replies", "Respond to at least 2 classmates (100 words each)")}</li>
        <li><strong>Due:</strong> {disc.get("due", "Initial post by Wednesday; replies by Sunday")}</li>
      </ul>
    </div>"""
        disc_html = _wrap_page(
            f"Week {week_num}: Discussion",
            course_code, week_num, disc_body
        )
        (week_dir / f"week_{week_num:02d}_discussion.html").write_text(disc_html, encoding="utf-8")

    # Count files generated
    files = list(week_dir.glob("*.html"))
    return len(files), [f.name for f in sorted(files)]


def generate_course(course_data_path: str, output_dir: str):
    """Generate a full course from a JSON data file."""
    data = json.loads(Path(course_data_path).read_text())
    out = Path(output_dir)
    course_code = data.get("course_code", "COURSE_101")

    total_files = 0
    for week in data["weeks"]:
        count, files = generate_week(week, out, course_code)
        total_files += count
        print(f"  Week {week['week_number']:2d}: {count} files - {', '.join(files)}")

    print(f"\nTotal: {total_files} files generated")
    return total_files


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_course.py <course_data.json> <output_dir>")
        sys.exit(1)
    generate_course(sys.argv[1], sys.argv[2])

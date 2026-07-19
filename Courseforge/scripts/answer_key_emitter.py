#!/usr/bin/env python3
"""Instructor answer-key emitter (assessment-quality overhaul, Phase 2 / C.2).

Deterministic, LLM-free renderer for the instructor-facing answer key that
carries everything the six IMS CC QTI 1.2 primitives cannot: per-item worked
solutions (each citing the source chunk ids the generator stamped),
per-distractor misconception rationale, and the deterministic Bloom-derived
rubric scaffolds for the essay / discussion / assignment surfaces (CC has no
portable machine-readable rubric).

Two products, written side-by-side into ``<export>/06_assessments/``:

  * ``answer_key.json`` — machine-readable, the authoritative copy.
  * ``answer_key.html`` — an accessible rendering packaged as a CC ``webcontent``
    resource (manifest ``assessments[]`` row ``type="answer_key"``).

**Grounding contract (anti-fabrication).** This module NEVER invents content.
It reshapes ONLY what the generator emitted on each ``QuestionData`` (read via
``Trainforge/generators/assessment_generator.py``): the correct answer(s), the
authored ``feedback`` (rendered as the worked-solution step body), the
per-distractor ``misconception_note`` on ``choices[]``, and the stamped
``source_chunks``. A field the generator did not populate renders as *explicitly
absent* ("not recorded" in HTML / an empty list in JSON) — never a fabricated
value. The rubric scaffolds are the one derived surface, and they are
deterministic structure keyed off the objective's Bloom level with the criterion
row citing the objective text — no content is authored.

Accepts both ``QuestionData`` dataclass instances AND their ``to_dict()`` form
(the checkpoint-replay path stores dicts), via attr-or-key accessors.

Portability: matching / ordering / error-analysis / two-tier pedagogy is
SIMULATED inside the CC primitives upstream; here it is surfaced honestly via
the ``item_subtype`` discriminator. No new QTI types are minted.
"""

from __future__ import annotations

import html as _html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA_VERSION = "v1"

# The numeric-FIB instructor note — CC carries no portable answer tolerance, so
# the instructor sets it in the LMS editor (spec Phase 3 / §B row 3).
_NUMERIC_TOLERANCE_NOTE = "Set the accepted-answer tolerance in your LMS editor."

# ``item_subtype`` tokens whose key is a computed numeric value (fill-in-blank).
_NUMERIC_SUBTYPES = frozenset({"fib_numeric"})

# Deterministic per-Bloom rubric focus (the criterion body derived from the
# objective's Bloom level). Structure only — no content-specific descriptors.
_BLOOM_RUBRIC_FOCUS: Dict[str, str] = {
    "remember": "Recalls the relevant facts, terms, and definitions accurately.",
    "understand": "Explains the concept in the learner's own words with correct meaning.",
    "apply": "Applies the correct procedure to the given situation and reaches a valid result.",
    "analyze": "Breaks the problem into its parts and examines the relationships between them.",
    "evaluate": "Justifies a judgement using appropriate criteria and supporting evidence.",
    "create": "Produces an original, well-structured response that integrates the ideas.",
}

# A fixed, generic 3-tier performance scale. The descriptors are deliberately
# generic scaffold labels (the specific expectation lives in the criterion text),
# so nothing content-specific is fabricated.
_RUBRIC_LEVELS: List[Dict[str, str]] = [
    {"level": "Proficient", "descriptor": "Fully meets the criterion."},
    {"level": "Developing", "descriptor": "Partially meets the criterion."},
    {"level": "Beginning", "descriptor": "Does not yet meet the criterion."},
]


# --------------------------------------------------------------------------- #
# attr-or-key accessors (accept dataclass OR to_dict() form)
# --------------------------------------------------------------------------- #
def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a dataclass attr OR a dict key; ``default`` on miss."""
    val = getattr(obj, name, None)
    if val is None and isinstance(obj, dict):
        val = obj.get(name)
    return default if val is None else val


def _strip_html(text: Any) -> str:
    """Strip tags to plain text (collapse whitespace). Never raises."""
    if not text:
        return ""
    plain = re.sub(r"<[^>]+>", " ", str(text))
    plain = _html.unescape(plain)
    return re.sub(r"\s+", " ", plain).strip()


def _as_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    return [str(value)]


# --------------------------------------------------------------------------- #
# per-item builders
# --------------------------------------------------------------------------- #
def _correct_answers_for(q: Any) -> List[str]:
    """Derive the correct-answer plain-text list, honouring the data model.

    Precedence: plural ``correct_answers`` (multiple-response) → scalar
    ``correct_answer`` → the ``is_correct`` choice texts. No fabrication: a
    key that is not present anywhere yields an empty list (rendered absent).
    """
    plural = _get(q, "correct_answers")
    if isinstance(plural, list) and plural:
        return [_strip_html(v) for v in plural]
    scalar = _get(q, "correct_answer")
    if scalar not in (None, ""):
        return [_strip_html(scalar)]
    keys: List[str] = []
    for c in _get(q, "choices", []) or []:
        if isinstance(c, dict) and c.get("is_correct"):
            keys.append(_strip_html(c.get("text")))
    return [k for k in keys if k]


def _worked_solution_steps(q: Any) -> List[Dict[str, Any]]:
    """Render the generator's authored ``feedback`` into worked-solution steps.

    Each step carries the SAME source chunk ids the generator stamped on the
    item (the grounding contract — solution steps cite the chunks the item was
    built from). A ``<ul>``/``<ol>`` feedback body splits into one step per
    ``<li>``; otherwise the whole feedback is one step. Absent feedback → an
    empty list (rendered "not recorded", never a fabricated solution).
    """
    feedback = _get(q, "feedback")
    if not feedback:
        return []
    cites = _as_str_list(_get(q, "source_chunks", []))
    raw = str(feedback)
    li_bodies = re.findall(r"<li[^>]*>(.*?)</li>", raw, flags=re.IGNORECASE | re.DOTALL)
    if li_bodies:
        texts = [_strip_html(b) for b in li_bodies]
    else:
        texts = [_strip_html(raw)]
    steps: List[Dict[str, Any]] = []
    for t in texts:
        if not t:
            continue
        steps.append({"text": t, "cites": list(cites)})
    return steps


def _per_distractor(q: Any) -> List[Dict[str, Any]]:
    """Collect the distractor choices + their targeted-misconception notes.

    Every ``is_correct == False`` choice is surfaced; its ``misconception_note``
    (when the generator authored one) becomes the "why wrong" rationale, else
    ``None`` (rendered "no misconception recorded" — honest absence).
    """
    rows: List[Dict[str, Any]] = []
    for c in _get(q, "choices", []) or []:
        if not isinstance(c, dict) or c.get("is_correct"):
            continue
        text = _strip_html(c.get("text"))
        if not text:
            continue
        note = c.get("misconception_note")
        rows.append({
            "choice": text,
            "misconception": _strip_html(note) if note else None,
        })
    return rows


def _bloom_rubric(bloom_level: Optional[str], objective_text: str) -> List[Dict[str, Any]]:
    """Deterministic rubric scaffold keyed off the objective's Bloom level.

    Three criterion rows: one cites the objective text verbatim, one is the
    Bloom-level focus, one is a generic clarity/organization row. Each carries
    the fixed 3-tier level scale. Structure only — no authored content.
    """
    level = (bloom_level or "").strip().lower()
    focus = _BLOOM_RUBRIC_FOCUS.get(level, _BLOOM_RUBRIC_FOCUS["understand"])
    obj = objective_text.strip() if objective_text else ""
    criterion_obj = (
        f"Response addresses the objective: {obj}" if obj
        else "Response addresses the stated objective."
    )
    return [
        {"criterion": criterion_obj, "levels": [dict(l) for l in _RUBRIC_LEVELS]},
        {"criterion": focus, "levels": [dict(l) for l in _RUBRIC_LEVELS]},
        {
            "criterion": "Response is clear, organized, and free of major errors.",
            "levels": [dict(l) for l in _RUBRIC_LEVELS],
        },
    ]


def _resolve_bloom_for_objective(
    objective_id: str,
    objective_bloom: Optional[Dict[str, str]],
    objective_text: str,
) -> Optional[str]:
    """Pick a Bloom level for a discussion/assignment rubric.

    Prefers an explicit per-objective Bloom map; else derives it from the
    objective statement via the deterministic verb ontology; else ``None``.
    """
    if objective_bloom and objective_id in objective_bloom:
        val = objective_bloom.get(objective_id)
        if val:
            return str(val)
    if objective_text:
        try:
            from lib.ontology.bloom import detect_bloom_level  # pure, offline
            level, _verb = detect_bloom_level(objective_text)
            if level:
                return level
        except Exception:  # noqa: BLE001 — never fail the key on a helper import
            return None
    return None


def _build_quiz_item(q: Any) -> Dict[str, Any]:
    """Reshape one ``QuestionData`` (or dict) into an answer-key item record."""
    q_type = str(_get(q, "question_type", ""))
    subtype = _get(q, "item_subtype")
    record: Dict[str, Any] = {
        "item_id": str(_get(q, "question_id", "")),
        "objective_id": str(_get(q, "objective_id", "")),
        "type": q_type,
        "item_subtype": str(subtype) if subtype else None,
        "correct_answers": _correct_answers_for(q),
        "worked_solution_steps": _worked_solution_steps(q),
        "per_distractor": _per_distractor(q),
        "source_chunk_ids": _as_str_list(_get(q, "source_chunks", [])),
    }
    linked = _get(q, "linked_item_id")
    if linked:
        record["linked_item_id"] = str(linked)
    # Numeric fill-in-blank rows stamp the LMS-tolerance instructor note (CC
    # carries no portable answer tolerance).
    if (subtype and str(subtype) in _NUMERIC_SUBTYPES) or q_type == "fill_in_blank":
        record["instructor_note"] = _NUMERIC_TOLERANCE_NOTE
    # Essay quiz items are hand-graded — attach a Bloom-derived rubric scaffold.
    if q_type == "essay":
        record["rubric"] = _bloom_rubric(
            _get(q, "bloom_level"), ""  # objective text lives on rubric_items
        )
    return record


# --------------------------------------------------------------------------- #
# top-level build
# --------------------------------------------------------------------------- #
def build_answer_key(
    *,
    assessments: Optional[List[Any]] = None,
    discussions: Optional[List[Dict[str, Any]]] = None,
    assignments: Optional[List[Dict[str, Any]]] = None,
    objective_text: Optional[Dict[str, str]] = None,
    objective_bloom: Optional[Dict[str, str]] = None,
    course_code: str = "",
    course_title: str = "",
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the ``answer_key.json`` document (pure — no I/O).

    ``assessments`` are ``AssessmentData`` instances or their ``to_dict()``
    form (each exposes ``questions``); ``discussions`` / ``assignments`` are the
    W10 prose payload dicts. ``objective_text`` maps objective ids → statement
    text for the rubric-criterion citation. All inputs optional → an empty but
    well-formed document.
    """
    assessments = assessments or []
    discussions = discussions or []
    assignments = assignments or []
    objective_text = objective_text or {}

    items: List[Dict[str, Any]] = []
    for a in assessments:
        for q in _get(a, "questions", []) or []:
            items.append(_build_quiz_item(q))

    rubric_items: List[Dict[str, Any]] = []
    for d in discussions:
        oid = str(d.get("objective_id") or "")
        obj_txt = objective_text.get(oid, "")
        bloom = _resolve_bloom_for_objective(oid, objective_bloom, obj_txt)
        rubric_items.append({
            "kind": "discussion",
            "id": str(d.get("discussion_id") or d.get("objective_id") or "discussion"),
            "objective_id": oid,
            "title": str(d.get("title") or "Discussion"),
            "bloom_level": bloom,
            "prompt": _strip_html(d.get("text")),
            "rubric": _bloom_rubric(bloom, obj_txt),
        })
    for t in assignments:
        oid = str(t.get("objective_id") or "")
        obj_txt = objective_text.get(oid, "")
        bloom = _resolve_bloom_for_objective(oid, objective_bloom, obj_txt)
        rubric_items.append({
            "kind": "assignment",
            "id": str(t.get("assignment_id") or t.get("objective_id") or "assignment"),
            "objective_id": oid,
            "title": str(t.get("title") or "Assignment"),
            "bloom_level": bloom,
            "task": _strip_html(t.get("text")),
            "rubric": _bloom_rubric(bloom, obj_txt),
        })

    doc: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "course_code": course_code,
        "title": course_title or (f"{course_code} — Instructor Answer Key"
                                  if course_code else "Instructor Answer Key"),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "item_count": len(items),
        "rubric_item_count": len(rubric_items),
        "items": items,
        "rubric_items": rubric_items,
    }
    return doc


# --------------------------------------------------------------------------- #
# accessible HTML rendering
# --------------------------------------------------------------------------- #
def _e(text: Any) -> str:
    return _html.escape(str(text if text is not None else ""))


def _render_rubric_table(rubric: List[Dict[str, Any]]) -> str:
    if not rubric:
        return "<p><em>No rubric recorded.</em></p>"
    parts: List[str] = [
        '<table class="rubric">',
        "<thead><tr>"
        '<th scope="col">Criterion</th>'
        '<th scope="col">Level</th>'
        '<th scope="col">Descriptor</th>'
        "</tr></thead>",
        "<tbody>",
    ]
    for row in rubric:
        levels = row.get("levels") or []
        span = max(1, len(levels))
        crit = _e(row.get("criterion"))
        if not levels:
            parts.append(
                f'<tr><th scope="row">{crit}</th><td>—</td><td>—</td></tr>'
            )
            continue
        for i, lv in enumerate(levels):
            if i == 0:
                parts.append(
                    f'<tr><th scope="row" rowspan="{span}">{crit}</th>'
                    f"<td>{_e(lv.get('level'))}</td>"
                    f"<td>{_e(lv.get('descriptor'))}</td></tr>"
                )
            else:
                parts.append(
                    f"<tr><td>{_e(lv.get('level'))}</td>"
                    f"<td>{_e(lv.get('descriptor'))}</td></tr>"
                )
    parts.append("</tbody></table>")
    return "".join(parts)


def _render_quiz_item(item: Dict[str, Any], idx: int) -> str:
    parts: List[str] = ['<article class="answer-item">']
    subtype = item.get("item_subtype")
    heading = f"Item {idx}: {_e(item.get('item_id'))}"
    parts.append(f"<h3>{heading}</h3>")
    meta: List[str] = [f"Type: {_e(item.get('type'))}"]
    if subtype:
        meta.append(f"Subtype: {_e(subtype)}")
    if item.get("objective_id"):
        meta.append(f"Objective: {_e(item.get('objective_id'))}")
    parts.append(f'<p class="meta">{" · ".join(meta)}</p>')

    # Correct answer(s).
    answers = item.get("correct_answers") or []
    if answers:
        if len(answers) == 1:
            parts.append(f"<p><strong>Correct answer:</strong> {_e(answers[0])}</p>")
        else:
            parts.append("<p><strong>Correct answers (select all):</strong></p>")
            parts.append("<ul>" + "".join(f"<li>{_e(a)}</li>" for a in answers) + "</ul>")
    else:
        parts.append("<p><strong>Correct answer:</strong> <em>not recorded</em></p>")

    if item.get("instructor_note"):
        parts.append(
            f'<p class="instructor-note"><strong>Instructor note:</strong> '
            f'{_e(item.get("instructor_note"))}</p>'
        )

    # Worked solution.
    steps = item.get("worked_solution_steps") or []
    parts.append("<h4>Worked solution</h4>")
    if steps:
        parts.append("<ol>")
        for st in steps:
            cites = st.get("cites") or []
            cite_txt = (
                f' <span class="cite">[source: {_e(", ".join(cites))}]</span>'
                if cites else ""
            )
            parts.append(f"<li>{_e(st.get('text'))}{cite_txt}</li>")
        parts.append("</ol>")
    else:
        parts.append("<p><em>No worked solution recorded.</em></p>")

    # Per-distractor rationale.
    distractors = item.get("per_distractor") or []
    if distractors:
        parts.append("<h4>Distractor rationale</h4>")
        parts.append(
            '<table class="distractors"><thead><tr>'
            '<th scope="col">Distractor</th>'
            '<th scope="col">Targeted misconception</th>'
            "</tr></thead><tbody>"
        )
        for d in distractors:
            mis = d.get("misconception")
            mis_cell = _e(mis) if mis else "<em>no misconception recorded</em>"
            parts.append(
                f'<tr><th scope="row">{_e(d.get("choice"))}</th>'
                f"<td>{mis_cell}</td></tr>"
            )
        parts.append("</tbody></table>")

    # Per-item rubric (essay).
    if item.get("rubric"):
        parts.append("<h4>Rubric</h4>")
        parts.append(_render_rubric_table(item["rubric"]))

    # Source grounding.
    chunks = item.get("source_chunk_ids") or []
    if chunks:
        parts.append(
            f'<p class="sources"><strong>Source chunks:</strong> '
            f'{_e(", ".join(chunks))}</p>'
        )
    parts.append("</article>")
    return "".join(parts)


def _render_rubric_item(item: Dict[str, Any], idx: int) -> str:
    parts: List[str] = ['<article class="answer-item">']
    kind = str(item.get("kind") or "task").title()
    parts.append(f"<h3>{kind} {idx}: {_e(item.get('title'))}</h3>")
    meta: List[str] = []
    if item.get("objective_id"):
        meta.append(f"Objective: {_e(item.get('objective_id'))}")
    if item.get("bloom_level"):
        meta.append(f"Bloom level: {_e(item.get('bloom_level'))}")
    if meta:
        parts.append(f'<p class="meta">{" · ".join(meta)}</p>')
    body = item.get("prompt") or item.get("task")
    if body:
        label = "Prompt" if item.get("kind") == "discussion" else "Task"
        parts.append(f"<p><strong>{label}:</strong> {_e(body)}</p>")
    parts.append("<h4>Rubric</h4>")
    parts.append(_render_rubric_table(item.get("rubric") or []))
    parts.append("</article>")
    return "".join(parts)


_ANSWER_KEY_CSS = """
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       max-width: 60rem; margin: 1rem auto; padding: 0 1rem; line-height: 1.5; }
h1 { border-bottom: 3px solid #2c5aa0; padding-bottom: .3rem; }
h2 { color: #2c5aa0; margin-top: 2rem; }
article.answer-item { border: 1px solid #e0e0e0; border-radius: 6px;
       padding: 1rem; margin: 1rem 0; }
p.meta { color: #444; font-size: .9rem; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0; }
th, td { border: 1px solid #e0e0e0; padding: .4rem .6rem; text-align: left;
         vertical-align: top; }
thead th { background: #f8f9fa; }
span.cite, p.sources { color: #555; font-size: .85rem; }
p.instructor-note { background: #f8f9fa; border-left: 4px solid #ffc107;
       padding: .4rem .6rem; }
"""


def render_answer_key_html(doc: Dict[str, Any]) -> str:
    """Render the answer-key document to an accessible standalone HTML page.

    Semantic headings, tables with ``scope``, no color-only signaling (every
    signal carries a text label). Escapes all interpolated content.
    """
    title = _e(doc.get("title") or "Instructor Answer Key")
    items = doc.get("items") or []
    rubric_items = doc.get("rubric_items") or []

    parts: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{title}</title>",
        f"<style>{_ANSWER_KEY_CSS}</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{title}</h1>",
        '<p role="note"><strong>Instructor copy.</strong> This answer key carries '
        "worked solutions, per-distractor rationale, and grading rubrics that the "
        "IMS Common Cartridge quiz format cannot transport. It is not intended for "
        "learners.</p>",
    ]

    if items:
        parts.append("<h2>Quiz items</h2>")
        for i, item in enumerate(items, start=1):
            parts.append(_render_quiz_item(item, i))
    if rubric_items:
        parts.append("<h2>Discussion &amp; assignment rubrics</h2>")
        for i, item in enumerate(rubric_items, start=1):
            parts.append(_render_rubric_item(item, i))
    if not items and not rubric_items:
        parts.append("<p><em>No graded assessment items were generated.</em></p>")

    parts.extend(["</main>", "</body>", "</html>"])
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# manifest row + one-call emit
# --------------------------------------------------------------------------- #
_ANSWER_KEY_HTML_NAME = "answer_key.html"
_ANSWER_KEY_JSON_NAME = "answer_key.json"


def answer_key_manifest_entry(*, title: str = "Instructor Answer Key") -> Dict[str, Any]:
    """The ``06_assessments/manifest.json`` ``assessments[]`` row for the key.

    ``file`` is the rendered HTML (the CC ``webcontent`` resource); ``type`` is
    the new ``answer_key`` discriminator the packager maps to ``webcontent``.
    """
    return {
        "file": _ANSWER_KEY_HTML_NAME,
        "type": "answer_key",
        "title": title,
        "identifier": "RES_answer_key",
    }


def emit_answer_key(
    out_dir: Path,
    *,
    assessments: Optional[List[Any]] = None,
    discussions: Optional[List[Dict[str, Any]]] = None,
    assignments: Optional[List[Dict[str, Any]]] = None,
    objective_text: Optional[Dict[str, str]] = None,
    objective_bloom: Optional[Dict[str, str]] = None,
    course_code: str = "",
    course_title: str = "",
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build + write ``answer_key.json`` and ``answer_key.html`` into ``out_dir``.

    Returns the manifest ``assessments[]`` row the caller appends to
    ``manifest.json``. PURE I/O over the deterministic builders above.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = build_answer_key(
        assessments=assessments,
        discussions=discussions,
        assignments=assignments,
        objective_text=objective_text,
        objective_bloom=objective_bloom,
        course_code=course_code,
        course_title=course_title,
        generated_at=generated_at,
    )
    (out_dir / _ANSWER_KEY_JSON_NAME).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / _ANSWER_KEY_HTML_NAME).write_text(
        render_answer_key_html(doc), encoding="utf-8"
    )
    return answer_key_manifest_entry(title=str(doc.get("title") or "Instructor Answer Key"))

"""Wave 77: Study-pack / lesson-plan renderer (pure SQL-over-metadata).

This module is the engine behind ``ed4all libv2 generate-study-pack``.
It assembles a single coherent document for a week (or set of weeks)
of a LibV2-archived course, ordering chunks by their editorial role:

    overview -> content_NN (numeric) -> application -> exercises ->
    self_check -> summary

No LLM is required: every input is structured metadata read from the
archive (``corpus/chunks.json`` + ``objectives.json``).

Output formats: ``md`` (markdown), ``html`` (standalone HTML5 page with
inline CSS), and ``json`` (structured payload for downstream tools).

The renderer is read-only against the archive; it never writes back.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html import escape as _html_escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------- #


# Canonical chunk_type ordering buckets used to assemble the pack.
# Lower bucket index sorts earlier in the final document.
_BUCKET_OVERVIEW = 0
_BUCKET_CONTENT = 1
_BUCKET_APPLICATION = 2
_BUCKET_EXERCISES = 3
_BUCKET_SELF_CHECK = 4
_BUCKET_SUMMARY = 5
_BUCKET_TRAILING = 6  # Anything we couldn't classify -> after summary.


# Canonical resource_type -> bucket map. ``resource_type`` lives in
# ``chunk.source.resource_type`` and reflects the editorial role of the
# enclosing module page.
_RESOURCE_TYPE_BUCKETS: Dict[str, int] = {
    "overview": _BUCKET_OVERVIEW,
    "page": _BUCKET_CONTENT,
    "content": _BUCKET_CONTENT,
    "application": _BUCKET_APPLICATION,
    "exercise": _BUCKET_EXERCISES,
    "exercises": _BUCKET_EXERCISES,
    "self_check": _BUCKET_SELF_CHECK,
    "self-check": _BUCKET_SELF_CHECK,
    "selfcheck": _BUCKET_SELF_CHECK,
    "quiz": _BUCKET_SELF_CHECK,
    "summary": _BUCKET_SUMMARY,
}


# Difficulty values we recognize, used to validate the --difficulty flag.
VALID_DIFFICULTIES = ("foundational", "intermediate", "advanced")


# Per-chunk timing rule: 2 minutes per 100 words, rounded to the
# nearest 5-minute increment, capped at 30 minutes per chunk.
_TIMING_MIN_PER_100_WORDS = 2.0
_TIMING_ROUND_TO = 5
_TIMING_CAP_MIN = 30


# ---------------------------------------------------------------------- #
# Errors
# ---------------------------------------------------------------------- #


class StudyPackError(RuntimeError):
    """Raised when a study pack cannot be rendered."""


# ---------------------------------------------------------------------- #
# Data classes
# ---------------------------------------------------------------------- #


@dataclass
class StudyPackChunk:
    """A single chunk projected for the study pack document."""

    chunk_id: str
    week: int
    chunk_type: str
    resource_type: str
    module_id: str
    bucket: int
    content_ordinal: int  # numeric ordinal within content_NN, 0 otherwise.
    position_in_module: int
    title: str
    text: str
    word_count: int
    difficulty: Optional[str]
    bloom_level: Optional[str]
    learning_outcome_refs: List[str]
    section_heading: Optional[str]
    source_references: List[Dict[str, Any]]

    @property
    def estimated_minutes(self) -> int:
        """Per-chunk timing estimate using the canonical formula."""
        if self.word_count <= 0:
            return 0
        raw = (float(self.word_count) / 100.0) * _TIMING_MIN_PER_100_WORDS
        rounded = int(round(raw / _TIMING_ROUND_TO) * _TIMING_ROUND_TO)
        if rounded == 0 and raw > 0:
            rounded = _TIMING_ROUND_TO
        return min(rounded, _TIMING_CAP_MIN)


@dataclass
class StudyPack:
    """Result of ``render_study_pack()``."""

    course_code: str
    course_title: str
    weeks: List[int]
    chunks: List[StudyPackChunk]
    objectives_referenced: List[Dict[str, Any]] = field(default_factory=list)
    assessment_chunks: List[StudyPackChunk] = field(default_factory=list)
    aggregated_source_references: List[Dict[str, Any]] = field(default_factory=list)
    lesson_plan_mode: bool = False

    @property
    def total_minutes(self) -> int:
        return sum(c.estimated_minutes for c in self.chunks)

    @property
    def total_words(self) -> int:
        return sum(c.word_count for c in self.chunks)


# ---------------------------------------------------------------------- #
# Archive reading helpers
# ---------------------------------------------------------------------- #


def _read_chunks(archive_root: Path) -> List[Dict[str, Any]]:
    """Read corpus/chunks.json (preferred) or chunks.jsonl as a fallback."""
    corpus_dir = archive_root / "corpus"
    json_path = corpus_dir / "chunks.json"
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StudyPackError(
                f"corpus/chunks.json is not valid JSON: {exc}"
            ) from exc
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
            return payload["chunks"]
        raise StudyPackError(
            "corpus/chunks.json has unexpected shape; expected a list "
            "or {'chunks': [...]}."
        )

    jsonl_path = corpus_dir / "chunks.jsonl"
    if jsonl_path.exists():
        chunks: List[Dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise StudyPackError(
                        f"corpus/chunks.jsonl: invalid JSON line: {exc}"
                    ) from exc
        return chunks

    raise StudyPackError(
        f"No corpus/chunks.json or corpus/chunks.jsonl under {archive_root}."
    )


def _read_objectives(archive_root: Path) -> Dict[str, Any]:
    """Read objectives.json. Empty dict if missing — non-fatal."""
    path = archive_root / "objectives.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StudyPackError(
            f"objectives.json is not valid JSON: {exc}"
        ) from exc


def _read_course(archive_root: Path) -> Dict[str, Any]:
    """Read course.json. Empty dict if missing — non-fatal."""
    path = archive_root / "course.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StudyPackError(
            f"course.json is not valid JSON: {exc}"
        ) from exc


# ---------------------------------------------------------------------- #
# Chunk classification
# ---------------------------------------------------------------------- #


_WEEK_RE = re.compile(r"^week_(\d+)")
_CONTENT_RE = re.compile(r"content[_-]?(\d+)", re.IGNORECASE)


def _parse_week(module_id: str) -> Optional[int]:
    if not module_id:
        return None
    match = _WEEK_RE.match(module_id)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _content_ordinal(module_id: str) -> int:
    """Extract the content_NN ordinal from a module_id; 0 when absent."""
    if not module_id:
        return 0
    match = _CONTENT_RE.search(module_id)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return 0


def _bucket_for(chunk_type: str, resource_type: str, module_id: str) -> int:
    """Decide which document bucket a chunk belongs to.

    Ordering rules (canonical):
        1. overview                  -> _BUCKET_OVERVIEW
        2. content_NN (numeric ord)  -> _BUCKET_CONTENT
        3. application               -> _BUCKET_APPLICATION
        4. exercise / exercises      -> _BUCKET_EXERCISES
        5. self_check / quiz         -> _BUCKET_SELF_CHECK
        6. summary                   -> _BUCKET_SUMMARY
    """
    rt = (resource_type or "").lower().strip()
    ct = (chunk_type or "").lower().strip()
    mid = (module_id or "").lower()

    # 1. Overview wins outright.
    if rt == "overview" or ct == "overview" or "overview" in mid:
        return _BUCKET_OVERVIEW

    # 5. Self-check / quiz module routing (resource_type or module slug).
    if rt in {"quiz", "self_check", "self-check", "selfcheck"} or "self_check" in mid:
        return _BUCKET_SELF_CHECK
    # assessment_item is the canonical chunk_type for self-check items.
    if ct == "assessment_item":
        return _BUCKET_SELF_CHECK

    # 6. Summary.
    if rt == "summary" or ct == "summary" or "summary" in mid:
        return _BUCKET_SUMMARY

    # 4. Exercises.
    if rt in {"exercise", "exercises"} or ct == "exercise" or "exercise" in mid:
        return _BUCKET_EXERCISES

    # 3. Application.
    if rt == "application" or "application" in mid:
        return _BUCKET_APPLICATION

    # 2. Content (page) — fallback bucket for non-special pages.
    if rt in {"page", "content"} or _CONTENT_RE.search(mid):
        return _BUCKET_CONTENT

    # Anything else parks at the end.
    return _BUCKET_TRAILING


def _project_chunk(raw: Dict[str, Any]) -> Optional[StudyPackChunk]:
    """Project a raw chunk dict into a StudyPackChunk; None if unsupported."""
    src = raw.get("source") or {}
    module_id = src.get("module_id") or ""
    week = _parse_week(module_id)
    if week is None:
        return None

    chunk_type = (raw.get("chunk_type") or "").strip()
    resource_type = (src.get("resource_type") or "").strip()
    bucket = _bucket_for(chunk_type, resource_type, module_id)
    content_ord = _content_ordinal(module_id)

    title = src.get("lesson_title") or src.get("module_title") or module_id
    title = _decode_html_entities(str(title))

    word_count_raw = raw.get("word_count")
    try:
        word_count = int(word_count_raw) if word_count_raw is not None else 0
    except (TypeError, ValueError):
        word_count = 0

    los = raw.get("learning_outcome_refs") or []
    if not isinstance(los, list):
        los = []
    los_clean = [str(x) for x in los if isinstance(x, (str, int))]

    src_refs = src.get("source_references") or []
    if not isinstance(src_refs, list):
        src_refs = []

    position = src.get("position_in_module")
    try:
        position_int = int(position) if position is not None else 0
    except (TypeError, ValueError):
        position_int = 0

    return StudyPackChunk(
        chunk_id=str(raw.get("id") or ""),
        week=week,
        chunk_type=chunk_type,
        resource_type=resource_type,
        module_id=module_id,
        bucket=bucket,
        content_ordinal=content_ord,
        position_in_module=position_int,
        title=title,
        text=str(raw.get("text") or "").strip(),
        word_count=word_count,
        difficulty=(raw.get("difficulty") or None),
        bloom_level=(raw.get("bloom_level") or None),
        learning_outcome_refs=los_clean,
        section_heading=(src.get("section_heading") or None),
        source_references=list(src_refs),
    )


_HTML_ENTITY_RE = re.compile(r"&(#x?[0-9A-Fa-f]+|[a-zA-Z]+);")
_NAMED_ENTITIES = {
    "mdash": "—",
    "ndash": "–",
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "hellip": "…",
    "rsquo": "’",
    "lsquo": "‘",
    "rdquo": "”",
    "ldquo": "“",
    "nbsp": " ",
}


def _decode_html_entities(value: str) -> str:
    """Decode the small set of HTML entities we see in lesson_title."""

    def _sub(match: "re.Match[str]") -> str:
        body = match.group(1)
        if body.startswith("#"):
            try:
                if body.startswith("#x") or body.startswith("#X"):
                    return chr(int(body[2:], 16))
                return chr(int(body[1:]))
            except (TypeError, ValueError):
                return match.group(0)
        return _NAMED_ENTITIES.get(body, match.group(0))

    return _HTML_ENTITY_RE.sub(_sub, value)


# ---------------------------------------------------------------------- #
# Sorting / filtering
# ---------------------------------------------------------------------- #


def _chunk_sort_key(c: StudyPackChunk) -> Tuple:
    """Canonical sort key for the assembled pack."""
    return (
        c.week,
        c.bucket,
        # Content_NN numeric ordinal next, then explicit position.
        c.content_ordinal,
        c.position_in_module,
        c.chunk_id,
    )


def _filter_chunks(
    chunks: Sequence[StudyPackChunk],
    *,
    weeks: Sequence[int],
    include_exercises: bool,
    include_self_check: bool,
    difficulties: Optional[Sequence[str]],
) -> List[StudyPackChunk]:
    """Apply opt-in inclusion + difficulty filters."""
    week_set = set(weeks)
    diff_set = (
        {d.lower().strip() for d in difficulties if d}
        if difficulties
        else None
    )

    out: List[StudyPackChunk] = []
    for c in chunks:
        if c.week not in week_set:
            continue
        if c.bucket == _BUCKET_EXERCISES and not include_exercises:
            continue
        if c.bucket == _BUCKET_SELF_CHECK and not include_self_check:
            continue
        if diff_set is not None:
            if (c.difficulty or "").lower() not in diff_set:
                continue
        out.append(c)
    return out


# ---------------------------------------------------------------------- #
# Objectives + sources aggregation
# ---------------------------------------------------------------------- #


def _objective_lookup(objectives: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build a case-insensitive id -> objective dict."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for to in (objectives.get("terminal_outcomes") or []):
        oid = str(to.get("id") or "").lower()
        if oid:
            lookup[oid] = to
    # Both legacy and current key names appear in the wild.
    for co in (
        objectives.get("component_objectives")
        or objectives.get("component_outcomes")
        or []
    ):
        oid = str(co.get("id") or "").lower()
        if oid:
            lookup[oid] = co
    return lookup


def _collect_objectives(
    chunks: Sequence[StudyPackChunk], objectives: Dict[str, Any]
) -> List[Dict[str, Any]]:
    lookup = _objective_lookup(objectives)
    seen: List[str] = []
    out: List[Dict[str, Any]] = []
    for c in chunks:
        for ref in c.learning_outcome_refs:
            key = ref.lower()
            if key in seen:
                continue
            seen.append(key)
            obj = lookup.get(key)
            if obj is None:
                # Pass through ID-only references when we can't resolve.
                out.append({"id": ref, "statement": "", "_unresolved": True})
            else:
                out.append({**obj, "id": obj.get("id") or ref})
    # Stable sort: TO before CO, then by id.
    def _rank(o: Dict[str, Any]) -> Tuple[int, str]:
        oid = str(o.get("id") or "").lower()
        return (0, oid) if oid.startswith("to-") else (1, oid)

    out.sort(key=_rank)
    return out


def _aggregate_sources(
    chunks: Sequence[StudyPackChunk],
) -> List[Dict[str, Any]]:
    seen_ids: set = set()
    out: List[Dict[str, Any]] = []
    for c in chunks:
        for ref in c.source_references:
            if not isinstance(ref, dict):
                continue
            sid = ref.get("sourceId")
            if not sid or sid in seen_ids:
                continue
            seen_ids.add(sid)
            out.append({k: v for k, v in ref.items()})
    out.sort(key=lambda r: str(r.get("sourceId") or ""))
    return out


# ---------------------------------------------------------------------- #
# Public API: render
# ---------------------------------------------------------------------- #


def render_study_pack(
    archive_root: Path,
    *,
    weeks: Sequence[int],
    include_exercises: bool = False,
    include_self_check: bool = False,
    difficulties: Optional[Sequence[str]] = None,
    lesson_plan: bool = False,
) -> StudyPack:
    """Read the archive, project + filter chunks, and return a StudyPack."""
    if not archive_root.exists():
        raise StudyPackError(f"Archive not found: {archive_root}")
    if not archive_root.is_dir():
        raise StudyPackError(f"Archive path is not a directory: {archive_root}")
    if not weeks:
        raise StudyPackError("At least one week must be specified.")

    raw_chunks = _read_chunks(archive_root)
    objectives = _read_objectives(archive_root)
    course = _read_course(archive_root)

    projected: List[StudyPackChunk] = []
    for raw in raw_chunks:
        proj = _project_chunk(raw)
        if proj is not None:
            projected.append(proj)

    filtered = _filter_chunks(
        projected,
        weeks=weeks,
        include_exercises=include_exercises,
        include_self_check=include_self_check,
        difficulties=difficulties,
    )
    filtered.sort(key=_chunk_sort_key)

    if not filtered:
        raise StudyPackError(
            f"No chunks found for week(s) {','.join(str(w) for w in weeks)} "
            f"under {archive_root} with the requested filters."
        )

    course_code = (
        course.get("course_code")
        or objectives.get("course_code")
        or _infer_course_code(filtered)
        or archive_root.name
    )
    course_title = course.get("title") or course.get("course_title") or course_code

    refs = _collect_objectives(filtered, objectives)
    sources = _aggregate_sources(filtered)
    assessments = [c for c in filtered if c.chunk_type == "assessment_item"]

    return StudyPack(
        course_code=str(course_code),
        course_title=str(course_title),
        weeks=sorted(set(weeks)),
        chunks=filtered,
        objectives_referenced=refs,
        assessment_chunks=assessments,
        aggregated_source_references=sources,
        lesson_plan_mode=lesson_plan,
    )


def _infer_course_code(chunks: Sequence[StudyPackChunk]) -> Optional[str]:
    for c in chunks:
        # source.course_id is set by the trainforge emitter.
        # (We didn't carry it on StudyPackChunk to keep the dataclass tight.)
        # Fall through: no source data on the projected chunk.
        return None
    return None


# ---------------------------------------------------------------------- #
# Markdown rendering
# ---------------------------------------------------------------------- #


_BUCKET_LABELS = {
    _BUCKET_OVERVIEW: "Overview",
    _BUCKET_CONTENT: "Core Content",
    _BUCKET_APPLICATION: "Application",
    _BUCKET_EXERCISES: "Exercises",
    _BUCKET_SELF_CHECK: "Self-Check",
    _BUCKET_SUMMARY: "Summary",
    _BUCKET_TRAILING: "Additional Material",
}


def _format_weeks(weeks: Sequence[int]) -> str:
    if len(weeks) == 1:
        return f"Week {weeks[0]}"
    return "Weeks " + ", ".join(str(w) for w in weeks)


def render_markdown(pack: StudyPack) -> str:
    lines: List[str] = []
    weeks_label = _format_weeks(pack.weeks)
    title_suffix = "Lesson Plan" if pack.lesson_plan_mode else "Study Pack"
    lines.append(f"# {pack.course_code}: {weeks_label} {title_suffix}")
    lines.append("")
    if pack.course_title and pack.course_title != pack.course_code:
        lines.append(f"_{pack.course_title}_")
        lines.append("")

    if pack.lesson_plan_mode:
        lines.append(
            f"**Total chunks:** {len(pack.chunks)}  "
            f"**Total words:** {pack.total_words:,}  "
            f"**Estimated time:** {pack.total_minutes} min"
        )
        lines.append("")
        lines.extend(_md_objectives_section(pack))
    else:
        lines.append(
            f"_{len(pack.chunks)} chunks "
            f"({pack.total_words:,} words)._"
        )
        lines.append("")

    current_bucket: Optional[int] = None
    for c in pack.chunks:
        if c.bucket != current_bucket:
            current_bucket = c.bucket
            lines.append(f"## {_BUCKET_LABELS.get(c.bucket, 'Section')}")
            lines.append("")
        lines.extend(_md_chunk_block(c, pack.lesson_plan_mode))

    if pack.lesson_plan_mode:
        if pack.assessment_chunks:
            lines.append("## Assessment Items")
            lines.append("")
            for c in pack.assessment_chunks:
                lines.append(f"### {c.title}")
                lines.append("")
                lines.append(c.text)
                lines.append("")
        if pack.aggregated_source_references:
            lines.append("## Resources")
            lines.append("")
            for ref in pack.aggregated_source_references:
                sid = ref.get("sourceId", "")
                role = ref.get("role", "")
                conf = ref.get("confidence")
                bits = [f"`{sid}`"]
                if role:
                    bits.append(f"role={role}")
                if conf is not None:
                    bits.append(f"confidence={conf}")
                lines.append(f"- {' '.join(bits)}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _md_objectives_section(pack: StudyPack) -> List[str]:
    if not pack.objectives_referenced:
        return []
    lines = ["## Learning Objectives", ""]
    lines.append("| ID | Bloom | Statement |")
    lines.append("|---|---|---|")
    for obj in pack.objectives_referenced:
        oid = str(obj.get("id") or "")
        bloom = str(obj.get("bloom_level") or "").strip() or "-"
        statement = str(obj.get("statement") or "").replace("|", "\\|")
        lines.append(f"| {oid.upper()} | {bloom} | {statement} |")
    lines.append("")
    return lines


def _md_chunk_block(c: StudyPackChunk, lesson_plan: bool) -> List[str]:
    lines: List[str] = []
    title = c.title or c.module_id or c.chunk_id
    lines.append(f"### {title}")
    lines.append("")
    if lesson_plan:
        meta_bits = [
            f"id: `{c.chunk_id}`",
            f"type: {c.chunk_type}",
            f"~{c.estimated_minutes} min",
            f"{c.word_count} words",
        ]
        if c.difficulty:
            meta_bits.append(f"difficulty: {c.difficulty}")
        if c.bloom_level:
            meta_bits.append(f"bloom: {c.bloom_level}")
        if c.learning_outcome_refs:
            meta_bits.append(
                "LOs: " + ", ".join(r.upper() for r in c.learning_outcome_refs)
            )
        lines.append("_" + " | ".join(meta_bits) + "_")
        lines.append("")

    if c.bucket == _BUCKET_EXERCISES:
        lines.append("```")
        lines.append(c.text)
        lines.append("```")
        lines.append("")
    elif c.bucket == _BUCKET_SELF_CHECK:
        # Markdown blockquote callout.
        for raw in c.text.splitlines():
            lines.append(f"> {raw}" if raw else ">")
        lines.append("")
    else:
        lines.append(c.text)
        lines.append("")
    return lines


# ---------------------------------------------------------------------- #
# HTML rendering
# ---------------------------------------------------------------------- #


_HTML_CSS = """\
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
max-width:48rem;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.55;}
h1{border-bottom:2px solid #444;padding-bottom:.4rem;}
h2{margin-top:2.2rem;color:#003a70;border-bottom:1px solid #cdd;padding-bottom:.2rem;}
h3{margin-top:1.6rem;color:#114a8a;}
.meta{color:#555;font-size:.92em;margin:.2rem 0 .8rem;}
.lo-table{border-collapse:collapse;width:100%;margin:.8rem 0;}
.lo-table th,.lo-table td{border:1px solid #cdd;padding:.4rem .6rem;
text-align:left;vertical-align:top;}
.lo-table th{background:#eef3fa;}
.exercise-block{background:#f6f8fa;border:1px solid #d0d7de;
border-radius:6px;padding:.8rem 1rem;font-family:Consolas,Menlo,monospace;
white-space:pre-wrap;}
.callout-self-check{border-left:4px solid #c08a00;background:#fff8e1;
padding:.6rem 1rem;margin:.6rem 0;}
.summary-bar{background:#eef3fa;padding:.4rem .8rem;border-radius:4px;
margin:.6rem 0 1rem;}
.resources li{font-family:Consolas,Menlo,monospace;font-size:.92em;}
"""


def render_html(pack: StudyPack) -> str:
    weeks_label = _format_weeks(pack.weeks)
    title_suffix = "Lesson Plan" if pack.lesson_plan_mode else "Study Pack"
    page_title = f"{pack.course_code}: {weeks_label} {title_suffix}"

    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head>')
    parts.append('<meta charset="utf-8" />')
    parts.append(f"<title>{_html_escape(page_title)}</title>")
    parts.append(f"<style>{_HTML_CSS}</style>")
    parts.append("</head><body>")
    parts.append(f"<h1>{_html_escape(page_title)}</h1>")
    if pack.course_title and pack.course_title != pack.course_code:
        parts.append(f"<p><em>{_html_escape(pack.course_title)}</em></p>")

    if pack.lesson_plan_mode:
        parts.append(
            '<div class="summary-bar">'
            f"<strong>Total chunks:</strong> {len(pack.chunks)}"
            f" &middot; <strong>Total words:</strong> {pack.total_words:,}"
            f" &middot; <strong>Estimated time:</strong> "
            f"{pack.total_minutes} min</div>"
        )
        parts.extend(_html_objectives_section(pack))
    else:
        parts.append(
            f"<p><em>{len(pack.chunks)} chunks "
            f"({pack.total_words:,} words).</em></p>"
        )

    current_bucket: Optional[int] = None
    for c in pack.chunks:
        if c.bucket != current_bucket:
            current_bucket = c.bucket
            label = _BUCKET_LABELS.get(c.bucket, "Section")
            parts.append(f"<h2>{_html_escape(label)}</h2>")
        parts.extend(_html_chunk_block(c, pack.lesson_plan_mode))

    if pack.lesson_plan_mode:
        if pack.assessment_chunks:
            parts.append("<h2>Assessment Items</h2>")
            for c in pack.assessment_chunks:
                parts.append(f"<h3>{_html_escape(c.title)}</h3>")
                parts.append(f"<div>{_paragraphize(c.text)}</div>")
        if pack.aggregated_source_references:
            parts.append("<h2>Resources</h2><ul class=\"resources\">")
            for ref in pack.aggregated_source_references:
                sid = _html_escape(str(ref.get("sourceId", "")))
                role = ref.get("role", "")
                conf = ref.get("confidence")
                bits = [sid]
                if role:
                    bits.append(f"role={_html_escape(str(role))}")
                if conf is not None:
                    bits.append(f"confidence={_html_escape(str(conf))}")
                parts.append("<li>" + " ".join(bits) + "</li>")
            parts.append("</ul>")

    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def _html_objectives_section(pack: StudyPack) -> List[str]:
    if not pack.objectives_referenced:
        return []
    parts = ["<h2>Learning Objectives</h2>"]
    parts.append('<table class="lo-table">')
    parts.append("<thead><tr><th>ID</th><th>Bloom</th><th>Statement</th></tr></thead>")
    parts.append("<tbody>")
    for obj in pack.objectives_referenced:
        oid = _html_escape(str(obj.get("id") or "").upper())
        bloom = _html_escape(str(obj.get("bloom_level") or "").strip() or "-")
        statement = _html_escape(str(obj.get("statement") or ""))
        parts.append(
            f"<tr><td>{oid}</td><td>{bloom}</td><td>{statement}</td></tr>"
        )
    parts.append("</tbody></table>")
    return parts


def _html_chunk_block(c: StudyPackChunk, lesson_plan: bool) -> List[str]:
    parts: List[str] = []
    title = _html_escape(c.title or c.module_id or c.chunk_id)
    parts.append(f"<h3>{title}</h3>")
    if lesson_plan:
        meta_bits = [
            f"id: <code>{_html_escape(c.chunk_id)}</code>",
            f"type: {_html_escape(c.chunk_type)}",
            f"~{c.estimated_minutes} min",
            f"{c.word_count} words",
        ]
        if c.difficulty:
            meta_bits.append(f"difficulty: {_html_escape(c.difficulty)}")
        if c.bloom_level:
            meta_bits.append(f"bloom: {_html_escape(c.bloom_level)}")
        if c.learning_outcome_refs:
            meta_bits.append(
                "LOs: "
                + ", ".join(_html_escape(r.upper()) for r in c.learning_outcome_refs)
            )
        parts.append('<p class="meta">' + " | ".join(meta_bits) + "</p>")

    if c.bucket == _BUCKET_EXERCISES:
        parts.append(
            f'<pre class="exercise-block">{_html_escape(c.text)}</pre>'
        )
    elif c.bucket == _BUCKET_SELF_CHECK:
        parts.append(
            f'<div class="callout-self-check">{_paragraphize(c.text)}</div>'
        )
    else:
        parts.append(f"<div>{_paragraphize(c.text)}</div>")
    return parts


def _paragraphize(text: str) -> str:
    """Split a text blob into <p> blocks on blank lines."""
    if not text:
        return ""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return "".join(
        f"<p>{_html_escape(p).replace(chr(10), '<br/>')}</p>"
        for p in paragraphs
        if p.strip()
    )


# ---------------------------------------------------------------------- #
# JSON rendering
# ---------------------------------------------------------------------- #


def render_json(pack: StudyPack) -> str:
    payload = {
        "course_code": pack.course_code,
        "course_title": pack.course_title,
        "weeks": pack.weeks,
        "lesson_plan_mode": pack.lesson_plan_mode,
        "total_chunks": len(pack.chunks),
        "total_words": pack.total_words,
        "total_minutes": pack.total_minutes,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "week": c.week,
                "chunk_type": c.chunk_type,
                "resource_type": c.resource_type,
                "module_id": c.module_id,
                "bucket": _BUCKET_LABELS.get(c.bucket, ""),
                "bucket_index": c.bucket,
                "content_ordinal": c.content_ordinal,
                "position_in_module": c.position_in_module,
                "title": c.title,
                "word_count": c.word_count,
                "estimated_minutes": c.estimated_minutes,
                "difficulty": c.difficulty,
                "bloom_level": c.bloom_level,
                "learning_outcome_refs": c.learning_outcome_refs,
                "section_heading": c.section_heading,
                "text": c.text,
            }
            for c in pack.chunks
        ],
    }
    if pack.lesson_plan_mode:
        payload["objectives"] = pack.objectives_referenced
        payload["assessment_chunk_ids"] = [c.chunk_id for c in pack.assessment_chunks]
        payload["resources"] = pack.aggregated_source_references
    return json.dumps(payload, indent=2) + "\n"


# ---------------------------------------------------------------------- #
# Top-level convenience
# ---------------------------------------------------------------------- #


def render(
    archive_root: Path,
    *,
    weeks: Sequence[int],
    output_format: str = "md",
    include_exercises: bool = False,
    include_self_check: bool = False,
    difficulties: Optional[Sequence[str]] = None,
    lesson_plan: bool = False,
) -> Tuple[StudyPack, str]:
    """Convenience wrapper: render a pack and serialize it in one call."""
    pack = render_study_pack(
        archive_root,
        weeks=weeks,
        include_exercises=include_exercises,
        include_self_check=include_self_check,
        difficulties=difficulties,
        lesson_plan=lesson_plan,
    )
    fmt = (output_format or "md").lower()
    if fmt == "md":
        return pack, render_markdown(pack)
    if fmt == "html":
        return pack, render_html(pack)
    if fmt == "json":
        return pack, render_json(pack)
    raise StudyPackError(
        f"Unknown output format: {output_format!r}. Expected md|html|json."
    )


__all__ = [
    "StudyPackError",
    "StudyPackChunk",
    "StudyPack",
    "render_study_pack",
    "render_markdown",
    "render_html",
    "render_json",
    "render",
    "VALID_DIFFICULTIES",
]

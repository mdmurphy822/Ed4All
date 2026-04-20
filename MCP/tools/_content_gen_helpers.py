"""Content-generation helpers for the textbook-to-course pipeline.

Supports ``MCP.tools.pipeline_tools._generate_course_content`` (Worker α) by:

- parsing staged DART HTML into a clean section/paragraph structure,
- synthesizing canonical learning-objective dicts (``CO-NN`` / ``TO-NN``)
  when no objectives JSON was supplied at pipeline entry,
- building per-week ``week_data`` payloads in the shape
  :func:`Courseforge.scripts.generate_course.generate_week` consumes.

The actual HTML rendering (full ``data-cf-*`` + JSON-LD surface) is delegated
to ``generate_week`` — Worker α is a thin orchestration wrapper that feeds
the mature Courseforge emitter with DART-derived inputs.

No external deps beyond the stdlib + existing Ed4All libraries.
"""

from __future__ import annotations

import html as _html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project imports — mature Bloom/taxonomy helpers.
from lib.ontology.bloom import detect_bloom_level
from lib.ontology.slugs import canonical_slug

# ---------------------------------------------------------------------------
# Constants and regexes
# ---------------------------------------------------------------------------

# Low-signal words to exclude when harvesting key terms from section text.
_STOPWORDS = frozenset([
    "the", "and", "that", "this", "with", "from", "have", "will", "they",
    "their", "these", "those", "such", "more", "most", "been", "also",
    "into", "over", "when", "what", "where", "which", "while", "than",
    "then", "them", "about", "among", "between", "both", "each", "some",
    "other", "because", "through", "across", "under", "upon", "every",
    "many", "only", "even", "just", "like", "here", "there",
    "your", "yours", "ours", "ourselves", "you", "we", "our",
    "its", "it's", "is", "are", "was", "were", "be", "been", "being",
    "has", "had", "do", "does", "did", "can", "could", "should", "would",
    "may", "might", "must", "shall", "who", "whom", "whose", "why", "how",
    "not", "no", "yes", "but", "for", "of", "in", "on", "at", "to", "by",
    "as", "an", "a", "or", "if", "so", "up", "out", "off", "per", "via",
])

# Section / heading boundary; DART emits both <h1> (page top) and <h2>/<h3>.
_SECTION_RE = re.compile(
    r"(?is)<section[^>]*>(.*?)(?=</section>|$)"
)
_HEADING_RE = re.compile(
    r"(?is)<(h[1-6])[^>]*>(.*?)</\1>"
)
_PARAGRAPH_RE = re.compile(r"(?is)<p[^>]*>(.*?)</p>")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Headings that almost certainly aren't real chapter/topic titles.
# Matched case-insensitively and via substring against the normalized heading
# (lowercased, whitespace-collapsed). Keeping this explicit rather than
# hand-tuning a classifier — catches the front-matter / back-matter /
# publisher-chrome categories that plagued the first real corpus run.
_HEADING_BLOCKLIST_TOKENS = frozenset([
    # Bibliographic / front / back matter
    "references", "bibliography", "index", "glossary", "appendix",
    "appendices", "abstract", "foreword", "preface", "afterword",
    "acknowledgements", "acknowledgments", "acknowledgement",
    "about the author", "about the authors", "about this book",
    "table of contents", "contents", "copyright", "isbn",
    # Publisher / location chrome
    "vancouver bc", "vancouver, bc", "toronto on", "london uk",
    "creative commons", "bccampus", "published by",
    # Generic / low-signal chapter chrome
    "purpose of the chapter", "purpose of this chapter",
    "overview", "introduction to this chapter", "in this chapter",
    "chapter summary", "chapter objectives", "key takeaways",
    "key points", "further reading", "further readings",
    "additional resources", "additional reading", "see also",
    "citation", "citations", "sources", "notes",
    # Navigation / template residue
    "skip to main content", "main content", "navigation",
    "table of", "learning objectives",
])

# Prefixes that mark a heading as a mid-sentence fragment that leaked into
# the `<h2>` parse (e.g. pdftotext misinterpretation of a running header).
_HEADING_LEADING_FRAGMENT_RE = re.compile(
    r"^(on the other hand|however|therefore|moreover|furthermore|"
    r"in addition|for example|for instance|that is|that said|"
    r"in contrast|similarly|as such|as a result|in summary|"
    r"in conclusion|winston churchill|once said)\b",
    re.IGNORECASE,
)

# City + 2-letter abbrev (VANCOUVER BC, LONDON UK, TORONTO ON, …).
_CITY_ABBREV_RE = re.compile(
    r"^[A-Z][A-Z ]{2,30}\s+[A-Z]{2}$"
)


def _is_low_signal_heading(heading: str) -> bool:
    """Return True when a heading looks like front/back-matter chrome,
    sentence-body residue, or otherwise isn't a real chapter/topic title.

    Applied inside ``parse_dart_html_files`` so downstream objective
    synthesis never turns publisher boilerplate or mid-paragraph
    fragments into a week topic.
    """
    if not heading:
        return True
    text = heading.strip()
    if not text:
        return True

    # Very short: single-word or bare-noun "title" — usually
    # TOC entries or running headers pulled by pdftotext.
    words = text.split()
    word_count = len(words)
    if word_count == 1 and len(text) <= 12:
        return True

    # Real chapter titles are rarely > 10 words. Anything longer is
    # almost always a sentence that pdftotext misread as a heading.
    if word_count > 10:
        return True

    # All-caps short heading — almost always chrome (e.g. VANCOUVER BC,
    # REFERENCES, ABSTRACT). Real chapter titles are usually Title Case
    # and ≥ 3 words.
    if text.isupper() and word_count <= 4 and len(text) <= 40:
        return True

    # City + 2-letter state/country pattern (VANCOUVER BC).
    if _CITY_ABBREV_RE.match(text):
        return True

    # Blocklist match (substring, case-insensitive).
    normalized = " ".join(text.lower().split())
    for token in _HEADING_BLOCKLIST_TOKENS:
        if token in normalized:
            return True

    # Looks like a mid-sentence fragment (lowercase start or
    # discourse-marker lead-in).
    if text[0].islower():
        return True
    if _HEADING_LEADING_FRAGMENT_RE.match(text):
        return True

    # Pure digits / numeric-metadata-looking (ISBN 978-..., page 42).
    if re.match(r"^\d+([.\-\s]\d+)*$", text):
        return True

    # Starts with an interrogative / conditional / adverbial starter
    # word that is almost never how a chapter title opens. ("Can you
    # imagine...", "If this book were offered...", "Thus there is a
    # continuum...", "Now add the metaproperty...")
    first_word = words[0].lower().rstrip(",.:;")
    if first_word in _SENTENCE_STARTER_WORDS:
        return True

    # Ends with a function word (preposition / conjunction / article) —
    # strong signal the heading is a truncated sentence. ("For my
    # personal comments on", "If this book were to be offered to a
    # commercial publisher, would you recommend it for")
    last_word = words[-1].lower().rstrip(",.:;!?")
    if last_word in _SENTENCE_TAIL_WORDS:
        return True

    # Mid-sentence period followed by more words → the heading spans
    # multiple sentences, which real titles don't.
    # ("Translational Research and the Semantic Web. Students should
    # study and")
    if re.search(r"\.\s+[A-Za-z]", text):
        return True

    # Error / log message residue from textbook code examples.
    if re.search(r"\b(an error occurred|exception|stack trace|"
                 r"traceback|undefined|not found|failed to)\b",
                 normalized):
        return True

    # Ends with a hyphen-truncated word fragment (pdftotext artifact
    # from text-body soft-hyphen line breaks that got misread as
    # heading). Example: "...have an inconsis-"
    if re.search(r"[A-Za-z]{2,}-$", text):
        return True

    # Repeated 2-word sequence within the heading — another pdftotext
    # artifact where a paragraph fragment double-parses. Example:
    # "AmountOfMatter and Living AmountOfMatter and Living have…"
    if word_count >= 4:
        lowered = [w.lower() for w in words]
        seen_pairs = set()
        for i in range(len(lowered) - 1):
            pair = (lowered[i], lowered[i + 1])
            if pair in seen_pairs:
                return True
            seen_pairs.add(pair)

    return False


# Words that real chapter titles almost never start with (but that
# show up as the first word when a sentence gets misread as a heading).
_SENTENCE_STARTER_WORDS = frozenset([
    "can", "could", "would", "should", "will", "does", "do", "did",
    "is", "are", "was", "were", "has", "have", "had",
    "what", "who", "whom", "whose", "which", "where", "when", "why",
    "how", "if", "unless", "because", "though", "although", "while",
    "now", "thus", "hence", "therefore", "moreover", "furthermore",
    "however", "nevertheless", "meanwhile", "still", "yet",
    "imagine", "consider", "note", "notice", "suppose", "assume",
    "given", "since", "so", "then", "also", "further",
])

# Function words that real chapter titles never end with.
_SENTENCE_TAIL_WORDS = frozenset([
    "and", "or", "but", "nor", "yet", "so",
    "of", "to", "for", "on", "at", "by", "in", "with", "as", "from",
    "into", "onto", "upon", "about", "against", "between", "through",
    "over", "under", "after", "before", "during",
    "the", "a", "an",
    "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "not", "no", "too",
])


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _strip_tags(fragment: str) -> str:
    """Strip HTML tags and decode entities; collapse whitespace."""
    text = _TAG_RE.sub(" ", fragment or "")
    text = _html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _extract_key_terms(text: str, max_terms: int = 4) -> List[str]:
    """Pick salient multi-word or capitalized terms from a block of text.

    Heuristic — no NLP deps. Prefers (a) multi-word capitalized phrases,
    then (b) repeated single words that are not stopwords. Returns up to
    ``max_terms`` display-cased terms (matching how a writer would key
    them); slugification happens downstream via ``canonical_slug``.
    """
    # (a) Capitalized bigrams / trigrams inside a sentence (not the first
    # word of a sentence — those are false positives).
    candidates: Dict[str, int] = {}
    # Sentence boundaries approximation: split on ". "
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = sentence.split()
        # Skip the first word — it's always capitalized at sentence start.
        for i in range(1, len(words)):
            w = words[i].strip(",.;:()[]\"'")
            if not w or not w[0].isupper():
                continue
            phrase_parts = [w]
            j = i + 1
            while j < len(words):
                nxt = words[j].strip(",.;:()[]\"'")
                if nxt and nxt[0].isupper() and nxt.lower() not in _STOPWORDS:
                    phrase_parts.append(nxt)
                    j += 1
                else:
                    break
            phrase = " ".join(phrase_parts)
            if len(phrase) < 4 or len(phrase) > 60:
                continue
            if phrase.lower() in _STOPWORDS:
                continue
            candidates[phrase] = candidates.get(phrase, 0) + 1

    # (b) Frequent standalone terms (length >= 5) — fallback only when
    # multi-word bigrams are sparse.
    if len(candidates) < max_terms:
        freq: Dict[str, int] = {}
        for word in re.findall(r"\b[A-Za-z][A-Za-z\-]{4,}\b", text):
            w = word.lower()
            if w in _STOPWORDS:
                continue
            freq[w] = freq.get(w, 0) + 1
        for w, count in sorted(freq.items(), key=lambda kv: -kv[1]):
            if count < 2:
                break
            display = w[0].upper() + w[1:]
            if display not in candidates:
                candidates[display] = count
            if len(candidates) >= max_terms:
                break

    ordered = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
    return [phrase for phrase, _ in ordered[:max_terms]]


# ---------------------------------------------------------------------------
# DART HTML parsing
# ---------------------------------------------------------------------------


def parse_dart_html_files(html_paths: List[Path]) -> List[Dict[str, Any]]:
    """Parse staged DART HTML files into a flat list of topic dicts.

    Each topic dict:
        {
            "heading": str,          # cleaned section heading
            "paragraphs": List[str], # cleaned paragraph text
            "key_terms": List[str],  # heuristic key terms (display case)
            "source_file": str,      # file stem for provenance
            "word_count": int,
        }

    Sections with < 30 words are skipped (usually metadata headers).
    """
    topics: List[Dict[str, Any]] = []
    for path in html_paths:
        try:
            html = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        stem = path.stem

        # Prefer DART-shaped <section> blocks; fall back to whole-document
        # heading-boundary split when no <section> tags are present.
        section_bodies = _SECTION_RE.findall(html)
        if not section_bodies:
            section_bodies = [html]

        for section_body in section_bodies:
            heading_match = _HEADING_RE.search(section_body)
            heading_raw = heading_match.group(2) if heading_match else ""
            heading = _strip_tags(heading_raw) or f"Section from {stem}"

            paragraphs_raw = _PARAGRAPH_RE.findall(section_body)
            paragraphs = []
            for para in paragraphs_raw:
                clean = _strip_tags(para)
                if len(clean) >= 40:
                    paragraphs.append(clean)

            if not paragraphs:
                continue

            full_text = " ".join(paragraphs)
            word_count = len(full_text.split())
            if word_count < 30:
                continue

            # Skip front/back-matter chrome and publisher boilerplate —
            # these leaked into the first real run (VANCOUVER BC,
            # REFERENCES, PURPOSE OF THE CHAPTER, mid-sentence
            # fragments) and turned whole weeks into junk.
            if _is_low_signal_heading(heading):
                continue

            topics.append({
                "heading": heading[:120],
                "paragraphs": paragraphs,
                "key_terms": _extract_key_terms(full_text),
                "source_file": stem,
                "word_count": word_count,
            })

    # De-duplicate topics that share a normalized heading (case- and
    # whitespace-insensitive). Keeps the first occurrence, which tends
    # to be the most content-rich one when a heading repeats across
    # front-matter / index / chapter body.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for topic in topics:
        key = " ".join(topic["heading"].lower().split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(topic)
    return deduped


def collect_staged_html(
    staging_dir: Optional[Path],
    inputs_root: Path,
) -> List[Path]:
    """Return the list of staged DART HTML files for this run.

    When ``staging_dir`` is a concrete directory, use it directly (the
    workflow runner passes this via ``phase_outputs``). Otherwise, fall
    back to scanning ``inputs_root`` (``Courseforge/inputs/textbooks/``)
    and picking the most recently modified run directory. Empty list
    when nothing is stageable — caller decides how to handle the miss.
    """
    candidate_files: List[Path] = []
    if staging_dir and staging_dir.exists() and staging_dir.is_dir():
        for f in sorted(staging_dir.iterdir()):
            if f.suffix.lower() in (".html", ".htm"):
                candidate_files.append(f)
    if candidate_files:
        return candidate_files

    if not inputs_root.exists():
        return []
    run_dirs = [d for d in inputs_root.iterdir() if d.is_dir()]
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        for f in sorted(run_dir.iterdir()):
            if f.suffix.lower() in (".html", ".htm"):
                candidate_files.append(f)
        if candidate_files:
            break
    return candidate_files


# ---------------------------------------------------------------------------
# Objective synthesis / normalization
# ---------------------------------------------------------------------------


def _normalize_objective_entry(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Coerce an objective dict (either camelCase or snake_case) to the
    shape ``generate_course.generate_week`` consumes.

    Dropped when the statement is missing or the ID doesn't fit the
    schema-enforced ``^[A-Z]{2,}-\\d{2,}$`` pattern — those entries would
    crash JSON-LD validation downstream, so we filter them here rather
    than emit broken pages.
    """
    if not isinstance(raw, dict):
        return None
    statement = raw.get("statement") or raw.get("description") or ""
    statement = statement.strip()
    if not statement:
        return None
    obj_id = raw.get("id") or raw.get("objective_id") or ""
    obj_id = obj_id.strip()
    if not re.match(r"^[A-Z]{2,}-\d{2,}$", obj_id):
        return None
    bloom_level = (
        raw.get("bloom_level")
        or raw.get("bloomLevel")
        or None
    )
    bloom_verb = (
        raw.get("bloom_verb")
        or raw.get("bloomVerb")
        or None
    )
    if not bloom_level:
        detected_level, detected_verb = detect_bloom_level(statement)
        bloom_level = bloom_level or detected_level
        bloom_verb = bloom_verb or detected_verb
    entry: Dict[str, Any] = {"id": obj_id, "statement": statement}
    if bloom_level:
        entry["bloom_level"] = bloom_level
    if bloom_verb:
        entry["bloom_verb"] = bloom_verb
    key_concepts = raw.get("key_concepts") or raw.get("keyConcepts")
    if key_concepts:
        entry["key_concepts"] = list(key_concepts)
    return entry


def load_objectives_json(
    objectives_path: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load terminal + chapter objectives from an objectives JSON file.

    Returns ``(terminal_objectives, chapter_objectives)``. Both lists are
    normalized to the generator shape; malformed entries are dropped.
    Empty lists when the file is missing, empty, or unreadable.
    """
    if not objectives_path:
        return ([], [])
    p = Path(objectives_path)
    if not p.exists():
        return ([], [])
    try:
        data = __import__("json").loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ([], [])

    terminal_raw = data.get("terminal_objectives", []) or []
    chapter_raw = data.get("chapter_objectives", []) or []

    # chapter_objectives may either be a flat list of objective dicts OR
    # a list of {"chapter": str, "objectives": [...]} groups. Flatten both.
    chapter_flat: List[Dict[str, Any]] = []
    for entry in chapter_raw:
        if not isinstance(entry, dict):
            continue
        if "objectives" in entry and isinstance(entry["objectives"], list):
            chapter_flat.extend(entry["objectives"])
        else:
            chapter_flat.append(entry)

    terminal = [
        e for e in (_normalize_objective_entry(o) for o in terminal_raw)
        if e is not None
    ]
    chapter = [
        e for e in (_normalize_objective_entry(o) for o in chapter_flat)
        if e is not None
    ]
    return (terminal, chapter)


def synthesize_objectives_from_topics(
    topics: List[Dict[str, Any]],
    duration_weeks: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate canonical objectives from parsed DART topics.

    Output shape matches what ``generate_week`` consumes. IDs follow the
    JSON-LD schema's ``^[A-Z]{2,}-\\d{2,}$`` pattern: ``TO-NN`` for terminal,
    ``CO-NN`` for chapter-level.

    One terminal objective per week of content (capped at ``duration_weeks``),
    plus two chapter objectives per week — one to introduce the material at
    ``understand`` level, one to apply it at ``analyze`` level. This keeps
    the 5-page-per-week emission schema-clean even when no objectives JSON
    was provided at pipeline entry.
    """
    if not topics:
        # Minimal scaffolding so downstream emission still produces valid
        # pages. No DART content, no real objectives — this is an edge
        # case for "empty corpus" runs.
        terminal = [{
            "id": "TO-01",
            "statement": "Summarize the foundational concepts covered by the course.",
            "bloom_level": "understand",
            "bloom_verb": "summarize",
            "key_concepts": [],
        }]
        chapter = [{
            "id": "CO-01",
            "statement": "Identify the main ideas introduced in the course materials.",
            "bloom_level": "remember",
            "bloom_verb": "identify",
            "key_concepts": [],
        }]
        return (terminal, chapter)

    # Group topics into weeks round-robin — later used for both objective
    # derivation and per-week content binding.
    topics_per_week = _group_topics_by_week(topics, duration_weeks)

    terminal: List[Dict[str, Any]] = []
    chapter: List[Dict[str, Any]] = []

    to_counter = 1
    co_counter = 1

    for week_num, week_topics in enumerate(topics_per_week, start=1):
        if not week_topics:
            continue
        primary = week_topics[0]
        primary_heading = primary["heading"]
        primary_terms = primary.get("key_terms") or [primary_heading]
        # One TO per week.
        terminal.append({
            "id": f"TO-{to_counter:02d}",
            "statement": (
                f"Apply concepts from {primary_heading.lower()} to analyze "
                f"real-world examples."
            ),
            "bloom_level": "apply",
            "bloom_verb": "apply",
            "key_concepts": [canonical_slug(t) for t in primary_terms[:3]
                             if canonical_slug(t)],
        })
        to_counter += 1

        # Two COs per week — one understand, one analyze.
        chapter.append({
            "id": f"CO-{co_counter:02d}",
            "statement": (
                f"Describe {primary_heading.lower()} and explain the core "
                f"ideas presented in the source material."
            ),
            "bloom_level": "understand",
            "bloom_verb": "describe",
            "key_concepts": [canonical_slug(t) for t in primary_terms[:3]
                             if canonical_slug(t)],
        })
        co_counter += 1

        if len(week_topics) > 1:
            secondary = week_topics[1]
            sec_terms = secondary.get("key_terms") or [secondary["heading"]]
            chapter.append({
                "id": f"CO-{co_counter:02d}",
                "statement": (
                    f"Differentiate key aspects of {secondary['heading'].lower()} "
                    f"and compare them with related concepts."
                ),
                "bloom_level": "analyze",
                "bloom_verb": "differentiate",
                "key_concepts": [canonical_slug(t) for t in sec_terms[:3]
                                 if canonical_slug(t)],
            })
            co_counter += 1

    return (terminal, chapter)


def _group_topics_by_week(
    topics: List[Dict[str, Any]],
    duration_weeks: int,
) -> List[List[Dict[str, Any]]]:
    """Return a list of length ``duration_weeks``; each entry is the list
    of topics assigned to that week.

    Distribution is block-based: consecutive topics stay together in the
    same week, which mirrors how a textbook's chapter ordering maps to a
    week sequence. Empty weeks are preserved so the caller can still emit
    the full 5-page template (with synthetic content) for them.
    """
    if duration_weeks <= 0:
        return []
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(duration_weeks)]
    if not topics:
        return buckets
    # Topics per week, rounded up.
    per_week = max(1, (len(topics) + duration_weeks - 1) // duration_weeks)
    for idx, topic in enumerate(topics):
        week_idx = min(idx // per_week, duration_weeks - 1)
        buckets[week_idx].append(topic)
    return buckets


# ---------------------------------------------------------------------------
# Per-week week_data assembly
# ---------------------------------------------------------------------------


def build_week_data(
    week_num: int,
    duration_weeks: int,
    week_topics: List[Dict[str, Any]],
    week_objectives: List[Dict[str, Any]],
    all_objectives: List[Dict[str, Any]],
    course_code: str,
) -> Dict[str, Any]:
    """Assemble the ``week_data`` dict that
    :func:`Courseforge.scripts.generate_course.generate_week` consumes.

    Shape reference: the fixture in
    ``tests/fixtures/pipeline/reference_week_01/``. We produce one
    ``overview`` page (from the first topic / week heading), one
    ``content`` module (from the second topic or a fallback synthesis),
    one ``application`` activity, one ``self_check`` quiz, and one
    ``summary`` — matching the contracts.md 5-page requirement.

    Per ``week_objectives`` is injected by the caller; this builder does
    NOT re-derive objectives (so callers can use canonical TO/CO IDs from
    an externally-supplied objectives JSON when available).
    """
    primary_topic = week_topics[0] if week_topics else None
    secondary_topic = week_topics[1] if len(week_topics) > 1 else primary_topic

    if primary_topic:
        week_title = primary_topic["heading"]
    else:
        week_title = f"Week {week_num} Concepts"

    # Overview: week-level paragraphs + readings
    overview_text: List[str] = []
    if primary_topic and primary_topic["paragraphs"]:
        overview_text.append(primary_topic["paragraphs"][0])
        if len(primary_topic["paragraphs"]) > 1:
            overview_text.append(primary_topic["paragraphs"][1])
    else:
        overview_text.append(
            f"This week surveys foundational concepts in {week_title.lower()}. "
            f"Work through the overview, content, application, self-check, and "
            f"summary pages in order."
        )

    # Content sections — emit one content page per week, with up to 2 sections.
    content_sections: List[Dict[str, Any]] = []
    if primary_topic:
        content_sections.append(
            _topic_to_section(primary_topic, section_role="definition"),
        )
    if secondary_topic and secondary_topic is not primary_topic:
        content_sections.append(
            _topic_to_section(secondary_topic, section_role="explanation"),
        )
    if not content_sections:
        content_sections.append({
            "heading": week_title,
            "level": 2,
            "content_type": "explanation",
            "paragraphs": [
                f"This week introduces the foundational ideas behind "
                f"{week_title.lower()}. Review the associated objectives and "
                f"note the key terms listed below."
            ],
            "key_terms": [],
        })

    content_modules = [{
        "title": primary_topic["heading"] if primary_topic else week_title,
        "sections": content_sections,
        "misconceptions": _build_misconceptions_for_week(week_topics),
    }]

    # Activities — one practice activity tied to first objective.
    activity_objective_ref = (
        week_objectives[0]["id"] if week_objectives else None
    )
    activities = [{
        "title": f"Apply: {week_title}",
        "description": _html.escape(
            f"Work through a scenario that applies the concepts from "
            f"{week_title.lower()} to a real-world example. Sketch a short "
            f"response (150 words) or diagram that shows how the key ideas "
            f"connect to practice."
        ),
        "bloom_level": "apply",
        **({"objective_ref": activity_objective_ref}
           if activity_objective_ref else {}),
    }]

    # Self-check questions — one question per objective (min 1).
    self_check_questions = _build_self_check_questions(
        week_topics, week_objectives
    )

    # Summary key takeaways.
    key_takeaways: List[str] = []
    if primary_topic:
        key_takeaways.append(
            f"{primary_topic['heading']}: review the definitions and core "
            f"relationships covered this week."
        )
    if secondary_topic and secondary_topic is not primary_topic:
        key_takeaways.append(
            f"{secondary_topic['heading']}: pay special attention to how "
            f"this connects with the earlier concepts."
        )
    key_takeaways.append(
        f"Map each learning objective in Week {week_num} to the sections "
        f"where it's introduced and practiced."
    )

    reflection_questions = [
        (
            f"Which idea from this week's material on "
            f"{week_title.lower()} challenged your prior understanding?"
        ),
        (
            "How would you explain the week's core concept to someone "
            "encountering it for the first time?"
        ),
    ]

    return {
        "week_number": week_num,
        "title": week_title,
        "estimated_hours": "3-4",
        "objectives": week_objectives or all_objectives[:2],
        "overview_text": overview_text,
        "content_modules": content_modules,
        "activities": activities,
        "self_check_questions": self_check_questions,
        "key_takeaways": key_takeaways,
        "reflection_questions": reflection_questions,
        "misconceptions": _build_misconceptions_for_week(week_topics),
    }


def _topic_to_section(
    topic: Dict[str, Any],
    section_role: str,
) -> Dict[str, Any]:
    """Convert a parsed topic dict to a ``generate_week`` section dict.

    Intentionally omits ``flip_cards`` — those trigger the mature
    emitter's ``teachingRole`` JSON-LD key which is not yet in
    ``courseforge_jsonld_v1.schema.json``. The Courseforge self-check /
    application pages still emit component metadata (via generate_week)
    because those keys live on the HTML elements, not the JSON-LD.
    """
    key_terms = topic.get("key_terms", []) or []
    paragraphs = [_html.escape(p) for p in topic["paragraphs"][:3]]
    return {
        "heading": topic["heading"],
        "level": 2,
        "content_type": section_role,
        "paragraphs": paragraphs,
        "key_terms": key_terms,
    }


def _build_self_check_questions(
    week_topics: List[Dict[str, Any]],
    week_objectives: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Produce 1–2 formative self-check questions for the week.

    Questions are multiple-choice with one correct option + two
    distractors. Wording is deterministic and topic-grounded so the
    self-check page validates against the JSON-LD schema and the
    page_objectives gate.
    """
    questions: List[Dict[str, Any]] = []
    focus_topic = week_topics[0] if week_topics else None
    focus_heading = focus_topic["heading"] if focus_topic else "this week's topic"
    primary_term = (
        focus_topic["key_terms"][0]
        if focus_topic and focus_topic.get("key_terms")
        else focus_heading
    )
    first_obj = week_objectives[0] if week_objectives else None
    q1: Dict[str, Any] = {
        "question": (
            f"Which of the following best describes the central idea of "
            f"{focus_heading.lower()}?"
        ),
        "bloom_level": "understand",
        "options": [
            {
                "text": (
                    f"A concept rooted in {primary_term.lower()} and "
                    f"related key terms covered this week."
                ),
                "correct": True,
                "feedback": (
                    f"Correct — {primary_term} is the central idea "
                    f"introduced in this week's reading."
                ),
            },
            {
                "text": "An unrelated topic reserved for later in the course.",
                "correct": False,
                "feedback": (
                    f"Not quite — revisit the overview page to see how "
                    f"{primary_term.lower()} anchors this week."
                ),
            },
            {
                "text": "A minor footnote only mentioned in passing.",
                "correct": False,
                "feedback": (
                    f"The material gives {primary_term.lower()} significant "
                    f"attention; re-read the content page for context."
                ),
            },
        ],
    }
    if first_obj:
        q1["objective_ref"] = first_obj["id"]
    questions.append(q1)

    if len(week_topics) > 1 and len(week_objectives) > 1:
        second_topic = week_topics[1]
        second_heading = second_topic["heading"]
        q2: Dict[str, Any] = {
            "question": (
                f"How does {second_heading.lower()} extend or complement "
                f"the earlier material this week?"
            ),
            "bloom_level": "analyze",
            "objective_ref": week_objectives[1]["id"],
            "options": [
                {
                    "text": (
                        f"It builds directly on the foundational ideas "
                        f"introduced earlier."
                    ),
                    "correct": True,
                    "feedback": (
                        f"Correct — the two topics reinforce each other in "
                        f"the week's sequence."
                    ),
                },
                {
                    "text": "It replaces the earlier material entirely.",
                    "correct": False,
                    "feedback": (
                        "The week's pages are sequential; later content "
                        "builds on — not replaces — earlier content."
                    ),
                },
                {
                    "text": "It is unrelated and only included for balance.",
                    "correct": False,
                    "feedback": (
                        "Both sections address the same week's theme; "
                        "revisit the overview."
                    ),
                },
            ],
        }
        questions.append(q2)
    return questions


def _build_misconceptions_for_week(
    week_topics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return a small set of synthesized misconceptions for the week.

    Kept short and deterministic — the ``misconception.json`` downstream
    artifact is Worker β's responsibility; here we just seed the JSON-LD
    ``misconceptions`` array so Trainforge can extract them.
    """
    if not week_topics:
        return []
    focus = week_topics[0]
    heading = focus["heading"]
    return [{
        "misconception": (
            f"Students often assume {heading.lower()} is a single idea "
            f"with a single definition."
        ),
        "correction": (
            f"In practice, {heading.lower()} encompasses several related "
            f"concepts — review the content page's key terms to see how "
            f"they connect."
        ),
    }]


# ---------------------------------------------------------------------------
# Page post-processing — objectives injection
# ---------------------------------------------------------------------------


# Regex to find the <h1> line inside <main id="main-content">; we inject the
# objectives block immediately after it. Every Courseforge page has this.
_H1_INSIDE_MAIN_RE = re.compile(
    r"(<main[^>]*id=\"main-content\"[^>]*>\s*<h1[^>]*>.*?</h1>)",
    re.DOTALL,
)

# Sentinel so we don't double-inject on re-emit.
_OBJECTIVES_SENTINEL = 'id="objectives"'


def _render_objectives_section(
    objectives: List[Dict[str, Any]],
) -> str:
    """Render a ``<section id="objectives">`` block with per-objective
    ``data-cf-objective-id`` / ``data-cf-bloom-*`` attributes.

    Mirrors the reference_week_01 fixture shape so every emitted page
    has a discoverable objectives surface (the ``page_objectives`` gate
    scans for ``data-cf-objective-id`` on every page, not just overview).
    """
    if not objectives:
        return ""
    items = []
    for obj in objectives:
        obj_id = obj.get("id", "")
        if not obj_id:
            continue
        statement = obj.get("statement", "")
        bloom_level = obj.get("bloom_level")
        bloom_verb = obj.get("bloom_verb")
        if not bloom_level:
            detected_level, detected_verb = detect_bloom_level(statement)
            bloom_level = bloom_level or detected_level
            bloom_verb = bloom_verb or detected_verb
        domain_map = {
            "remember": "factual",
            "understand": "conceptual",
            "apply": "procedural",
            "analyze": "conceptual",
            "evaluate": "metacognitive",
            "create": "procedural",
        }
        domain = domain_map.get(bloom_level or "", "conceptual")
        attrs = [f'data-cf-objective-id="{_html.escape(obj_id)}"']
        if bloom_level:
            attrs.append(f'data-cf-bloom-level="{bloom_level}"')
        if bloom_verb:
            attrs.append(f'data-cf-bloom-verb="{_html.escape(bloom_verb)}"')
        if domain:
            attrs.append(f'data-cf-cognitive-domain="{domain}"')
        items.append(
            f'      <li {" ".join(attrs)}>'
            f'<strong>{_html.escape(obj_id)}:</strong> '
            f'{_html.escape(statement)}</li>'
        )
    items_html = "\n".join(items)
    return (
        '\n    <section id="objectives" class="objectives" '
        'aria-labelledby="objectives-heading">\n'
        '      <h2 id="objectives-heading" data-cf-content-type="overview">'
        'Learning Objectives</h2>\n'
        '      <ul>\n'
        f'{items_html}\n'
        '      </ul>\n'
        '    </section>'
    )


def ensure_objectives_on_page(
    html_text: str,
    objectives: List[Dict[str, Any]],
) -> str:
    """Inject an objectives ``<section>`` block into a page when absent.

    Keeps the overview page's pre-existing ``.objectives`` block intact
    (skip-path via sentinel). For every other page that lacks objectives
    metadata, insert the block right after the page's ``<h1>``. Needed
    so Wave 2's ``page_objectives`` gate + the integration test's per-page
    ``data-cf-objective-id`` check both pass across all 5 pages.
    """
    if _OBJECTIVES_SENTINEL in html_text:
        return html_text
    if "data-cf-objective-id" in html_text:
        return html_text
    section = _render_objectives_section(objectives)
    if not section:
        return html_text
    return _H1_INSIDE_MAIN_RE.sub(
        lambda m: m.group(1) + section,
        html_text,
        count=1,
    )


__all__ = [
    "parse_dart_html_files",
    "collect_staged_html",
    "load_objectives_json",
    "synthesize_objectives_from_topics",
    "build_week_data",
    "ensure_objectives_on_page",
]

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
from lib.ontology.learning_objectives import mint_lo_id
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
# Wave 27: we capture the full opening tag too so we can harvest
# ``data-dart-block-id`` for source-id carry-through.
_SECTION_RE = re.compile(
    r"(?is)(<section[^>]*>)(.*?)(?=</section>|$)"
)
_HEADING_RE = re.compile(
    r"(?is)<(h[1-6])[^>]*>(.*?)</\1>"
)
# Wave 35: full-document heading fallback — we need both the start offset
# and the tag name so the scan-by-boundary pass can tie each paragraph
# back to the nearest enclosing <h2>/<h3>. ``_HEADING_BOUNDARY_RE``
# matches the opening tag only; the closing tag's offset is derived.
_HEADING_OPEN_RE = re.compile(
    r"(?is)<h([2-3])[^>]*>(.*?)</h\1>"
)
_PARAGRAPH_RE = re.compile(r"(?is)<p[^>]*>(.*?)</p>")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Wave 27: extract DART block-id from the section wrapper opening tag.
# DART Wave 8+ stamps ``data-dart-block-id="{block_id}"`` on every top-
# level section; we carry it through onto the Courseforge page's section
# element as ``data-cf-source-ids="dart:{slug}#{block_id}"``.
_DATA_DART_BLOCK_ID_RE = re.compile(
    r"""(?is)data-dart-block-id\s*=\s*["']([^"']+)["']"""
)

# Wave 24: DART Wave 13+ emits each chapter as <article role="doc-chapter">.
# When present, parse_dart_html_files tags every topic with its chapter_id so
# _group_topics_by_week can respect chapter boundaries when distributing
# topics across weeks.
_DOC_CHAPTER_ARTICLE_RE = re.compile(
    r"(?is)<article\s+[^>]*?role\s*=\s*[\"']doc-chapter[\"'][^>]*>(.*?)</article>"
)
_ARTICLE_ID_RE = re.compile(
    r"(?is)<article\s+[^>]*?id\s*=\s*[\"']([^\"']+)[\"']"
)

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
    # Wave 27 HIGH-4: publisher-credit / cover / copyright chrome that
    # has been observed leaking into week titles on real corpora.
    "cover design", "cover art", "designed by", "illustrated by",
    "illustrations by", "illustration by", "translation by",
    "all rights reserved", "rights reserved", "edited by",
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

# Author byline: a sequence of 2+ Title-Case tokens that look like names.
# Catches "A.B. Smith Jane Doe" / "Cover design by Author Name"
# — the tokens are all name-like (each starts with a capital, often hyphenated)
# with no verb, topical noun, or connector word. Discriminator is that the
# whole string is just proper nouns (and maybe the lead-ins "by" / "edited by"
# / "cover design by").
#
# Wave 27 broadens to include initialed names (e.g. single-letter initials
# like "J.", multi-initial sequences like "J.R.R.", titles like "Dr.") and
# parenthetical nicknames (e.g. "J.R.R. (Ronald) Tolkien"). Token may be
# all-caps when short (acronym initials) but a plain capitalized word
# (>= 2 letters) is still the common case.
_NAME_TOKEN_RE = re.compile(
    r"^(?:"
    r"[A-Z]\."                      # single initial: "A."
    r"|[A-Z](?:\.[A-Z])+\.?"        # multi initial: "A.W." / "J.R.R."
    r"|[A-Z][A-Za-z'\u00C0-\u017F\-]+"  # capitalized word, allows unicode diacritics + hyphens
    r"|\([A-Z][A-Za-z'\u00C0-\u017F\-]+\)"  # parenthetical nickname: "(Tony)"
    r")$"
)
_AUTHOR_BYLINE_LEADINS = frozenset([
    "by", "edited by", "foreword by", "preface by",
    "cover design by", "cover art by", "designed by",
    "illustrations by", "illustration by", "translation by",
])

# Wave 27 HIGH-4: formulaic-phrase markers. "The functional syntax
# equivalent is as follows:" style lead-ins are chapter-body prose
# erroneously promoted to section headings by pdftotext.
_FORMULAIC_PHRASE_RE = re.compile(
    r"(?i)(functional|logical|mathematical|formal)\s+"
    r"(syntax|form|expression|representation|notation|equivalent)"
)

# Wave 27 HIGH-4: math / logic notation detector. Any one of these Unicode
# symbols inside a short (≤ 40 chars) heading marks it as formula residue
# rather than a real chapter title. Used for cases like "C v ∀R.D",
# "∀x (P(x) → Q(x))", etc. Formulas are valid inline content but should
# never be treated as topic headings for objective synthesis.
_MATH_NOTATION_CHARS = frozenset(
    "∀∃∈∉⊆⊇∪∩∧∨¬⊤⊥≡≢⇒⇔→←↔⊃⊂≤≥≠≈"
    "∑∏∫∞∂∇αβγδεζηθικλμνξοπρστυφχψω"
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
)

# A heading that ends with a colon (`:`) is almost always a prompt /
# section-preamble rather than a real title. The exception is when the
# colon is followed by a short noun-phrase subtitle separated by a newline
# or em-dash — in that case the colon is a title:subtitle separator. This
# regex identifies the "bare prompt" shape: ends with ":" and has NO
# subtitle attached.
_COLON_PROMPT_TAIL_RE = re.compile(r":\s*$")

# Formula / notation fragments (e.g. "C v ∀R.D", "FirstYearCourse
# SubClassOf isTaughtBy only Professor"). These look unlike prose headings
# (unusual symbols / CamelCase multi-token strings) but for an ontology or
# formal-methods textbook they're pedagogically meaningful — real examples
# from the chapter body shown as section anchors. We KEEP them as valid
# headings. Detection is informational only; no rejection.
#
# Decision (recorded here per task directive): formula-like heading
# fragments are legitimate content in ontology / formal-methods textbooks
# and should be preserved even though they superficially look like
# sentence-body residue.


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

    # End-colon prompt heading ("This chapter covers the following topics:" /
    # "The functional syntax equivalent is as follows:"). These aren't
    # titles — they're the lead-in sentence to a list. Reject UNLESS the
    # heading is very short (≤ 3 words) and looks like a title:subtitle
    # prefix (e.g. "Introduction:" as a sole word).
    if _COLON_PROMPT_TAIL_RE.search(text) and word_count > 3:
        return True

    # Wave 27 HIGH-4: math / logic notation detector. Short heading
    # containing Unicode math symbols (∀ ∃ ∈ ⊆ ∧ ¬ etc.) is formula
    # residue from the chapter body, not a real chapter title. Applied
    # BEFORE the formulaic-phrase check because formula residues are
    # often all-symbols with no English word at all.
    if len(text) <= 40 and any(ch in _MATH_NOTATION_CHARS for ch in text):
        return True

    # Wave 27 HIGH-4: formulaic-phrase lead-ins that pdftotext hoisted
    # into a heading ("The functional syntax equivalent is as follows:").
    if _FORMULAIC_PHRASE_RE.search(text):
        return True

    # Author byline detector. A heading is a byline when:
    #   (1) the heading starts with an explicit byline lead-in
    #       ("by", "edited by", "cover design by") followed by 1+ Name-like
    #       tokens — this is a high-precision signal regardless of count
    #       ("Cover design by Author Name" is a byline even at two authors).
    #   (2) every token is a Name-like token AND at least one token is
    #       hyphenated / multi-initialed (high-confidence author signal
    #       like "J.K." or "A.B."). OR
    #   (3) Wave 27: exactly 2-3 tokens AND every token looks like a
    #       proper name AND no token matches the common-title-word set.
    #       Catches bare 2-name bylines ("Author Surname") without false-
    #       positive-demoting "European Union", "Creative Commons",
    #       "Digital Pedagogy", etc.
    if word_count >= 2:
        stripped_words = [w.strip(",.;:()[]\"'") for w in words]
        tokens_for_name_check = stripped_words
        leadin_matched = False
        lowered_joined = " ".join(w.lower() for w in stripped_words)
        for leadin in _AUTHOR_BYLINE_LEADINS:
            if lowered_joined.startswith(leadin + " "):
                leadin_words = leadin.split()
                tokens_for_name_check = stripped_words[len(leadin_words):]
                leadin_matched = True
                break
        if tokens_for_name_check and len(tokens_for_name_check) >= 1:
            all_name_like = all(
                _NAME_TOKEN_RE.match(tok) is not None
                and tok.lower() not in _STOPWORDS
                and tok.lower() not in _SENTENCE_STARTER_WORDS
                and tok.lower() not in _SENTENCE_TAIL_WORDS
                for tok in tokens_for_name_check
            )
            if all_name_like:
                hyphenated = any("-" in tok for tok in tokens_for_name_check)
                initialed = any(
                    "." in tok or (tok.startswith("(") and tok.endswith(")"))
                    for tok in tokens_for_name_check
                )
                # (1) lead-in → high-confidence byline regardless of count.
                # (2) hyphenated / initialed / parenthetical → strong name signal.
                # (3) pure 2-3 token capitalized sequence where NO token is
                #     in the curated common-title-word set (catches "Jane
                #     Doe" without tripping "European Union Policy").
                if leadin_matched or hyphenated or initialed:
                    return True
                if 2 <= len(tokens_for_name_check) <= 3:
                    any_common_title_word = any(
                        tok.lower() in _COMMON_TITLE_WORDS
                        for tok in tokens_for_name_check
                    )
                    if not any_common_title_word:
                        return True

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

# Wave 27 HIGH-4: common English title-bearing nouns / adjectives. When a
# short 2-3 token capitalized heading contains any of these, it's very likely
# a legitimate chapter / section title ("European Union Policy", "Digital
# Pedagogy", "Creative Commons", "Research Methods", "Science of Learning"),
# NOT a bare author byline ("Jane Doe", "John Smith"). Kept intentionally
# small and conservative — only adds a token here when its surname usage is
# rare AND it's a common title vocabulary word.
_COMMON_TITLE_WORDS = frozenset([
    # Disciplines / fields
    "science", "sciences", "research", "studies", "theory", "theories",
    "methods", "methodology", "analysis", "synthesis", "practice",
    "education", "learning", "teaching", "pedagogy", "psychology",
    "sociology", "philosophy", "economics", "mathematics", "physics",
    "chemistry", "biology", "computing", "engineering", "medicine",
    "history", "literature", "linguistics", "statistics", "genetics",
    "ethics", "aesthetics", "politics", "government", "policy", "policies",
    "law", "management", "leadership", "innovation", "technology",
    # Descriptors / modifiers
    "introduction", "overview", "foundations", "fundamentals", "principles",
    "advanced", "basic", "modern", "classical", "contemporary", "digital",
    "global", "international", "national", "regional", "local", "public",
    "private", "creative", "critical", "applied", "theoretical",
    "practical", "professional", "academic", "scientific", "european",
    "american", "asian", "african", "eastern", "western", "northern",
    "southern", "united",
    # Structural / nouns common in titles
    "chapter", "section", "module", "unit", "course", "program", "curriculum",
    "system", "systems", "design", "framework", "approach", "model",
    "perspective", "perspectives", "concept", "concepts", "process",
    "processes", "development", "assessment", "evaluation", "reform",
    "change", "growth", "world", "society", "community", "communities",
    "culture", "cultures", "environment", "environments", "institution",
    "institutions", "organization", "organizations", "movement", "movements",
    "revolution", "revolutions", "tradition", "traditions", "union",
    "commons", "commonwealth", "federation", "republic", "kingdom",
    "empire", "age", "era", "century", "past", "future", "present",
    "knowledge", "skills", "information", "communication", "media",
    "networks", "data", "health", "welfare", "justice", "rights",
    "democracy", "capitalism", "socialism", "liberalism", "conservatism",
    "feminism", "philosophy", "ontology", "epistemology", "logic",
    "logics", "reasoning", "representation", "computation", "cognition",
    "perception", "memory", "language", "grammar", "syntax", "semantics",
    "pragmatics", "phonetics", "morphology", "discourse",
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
# Source-corpus extractors (LOs / misconceptions / self-check questions)
# ---------------------------------------------------------------------------
#
# All three extractors are pure string processing over the DART-staged HTML
# text. They return empty lists when the source doesn't contain real entries
# of the given kind — per the content-generation policy, downstream pages
# should render with whatever real content exists and omit the rest rather
# than synthesize template prose.


# Headings that precede a learning-objectives bullet list.
_LO_HEADING_HINT_RE = re.compile(
    r"(?i)\b("
    r"learning\s+objectives?"
    r"|chapter\s+objectives?"
    r"|after\s+(?:reading|completing|studying)\s+this\s+(?:chapter|section|module|unit)"
    r"|by\s+the\s+end\s+of\s+this\s+(?:chapter|section|module|unit)"
    r"|students?\s+will\s+(?:be\s+able\s+to|learn|understand)"
    r"|you\s+will\s+be\s+able\s+to"
    r"|in\s+this\s+(?:chapter|section|module|unit)\s+you\s+will"
    r")\b"
)

# Inline prose lead-ins for LO sentences (when the chapter doesn't use a
# bullet list but writes "After reading this chapter you will be able to
# describe, explain, and compare …").
_LO_INLINE_LEADIN_RE = re.compile(
    r"(?is)(?:after\s+reading\s+this\s+chapter|by\s+the\s+end\s+of\s+this\s+(?:chapter|section)|"
    r"you\s+will\s+be\s+able\s+to|students?\s+will\s+be\s+able\s+to)"
    r"[,:]?\s*([^.]+?)\."
)

# Misconception extraction patterns.
# Two common shapes:
#   (a) "Misconception: ...\nCorrection: ..."
#   (b) "Common misconception: ..." / "A common mistake is ..." / "Students
#       often think …. In fact …"
_MISCONCEPTION_CORRECTION_PAIR_RE = re.compile(
    r"(?is)misconception\s*:\s*(.+?)"
    r"\s*correction\s*:\s*(.+?)"
    r"(?=\s*misconception\s*:|\s*$)"
)
_MISCONCEPTION_STANDALONE_RE = re.compile(
    r"(?i)(?:a\s+)?common\s+(?:misconception|mistake|error|misunderstanding)\s+"
    r"(?:is|here\s+is)\s*:?\s*(.+?)(?:\.\s|\.$)"
)

# "Warning:" / "Note that" / "Caution:" patterns typically flag things students
# get wrong. We capture just the statement after the marker.
_PITFALL_MARKER_RE = re.compile(
    r"(?i)(?:warning|caution|note\s+that|pitfall|beware)\s*:?\s+(.+?)(?:\.\s|\.$)"
)

# Self-check / exercise / review-question extraction.
# Textbook-style exercise markers: "Review question 2.3.", "Exercise 5.1.",
# "Self-check questions", "Activity 3".
_EXERCISE_MARKER_RE = re.compile(
    r"(?i)\b(?:review\s+question|exercise|self[-\s]check\s+question|activity|"
    r"practice\s+question|check\s+your\s+understanding)"
    r"\s+(\d+(?:\.\d+)*)"
    r"\s*[:.\-]?\s*(.+?)(?=\.\s+[A-Z]|\.$|\?|\n\n)"
)


def _split_sentences(text: str) -> List[str]:
    """Split a chunk of prose into sentences. Naive ``. `` boundary — fine
    for the extractors below which are themselves approximate."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def extract_learning_objectives(full_text: str) -> List[str]:
    """Return a list of learning-objective statements extracted from the
    source text. Each entry is a cleaned sentence (no leading bullet /
    numbering). Returns an empty list when no recognizable LO section was
    found — callers MUST NOT fabricate objectives from the heading or
    first paragraph when this returns [].

    Heuristic: locate any sentence / list block introduced by an LO header
    hint ("Learning Objectives", "After reading this chapter you will be
    able to…", etc.) and collect the items that immediately follow.
    """
    if not full_text:
        return []
    los: List[str] = []

    # Strategy 1: find an LO heading hint and harvest the items that follow.
    # We look for the hint anywhere in the text and take up to ~500 chars
    # after it as the LO "block"; split on newline / semicolon / bullet marks
    # and keep items that start with a Bloom-ish verb (or any normal verb).
    for m in _LO_HEADING_HINT_RE.finditer(full_text):
        tail = full_text[m.end(): m.end() + 800]
        # Stop at the next heading-ish marker (double newline, another LO
        # hint, or a hard sentence-ending heading like "Introduction").
        stop = re.search(r"\n\s*\n|Introduction\b|Summary\b", tail)
        if stop:
            tail = tail[: stop.start()]
        for item in re.split(r"[•\u2022\n;]|(?<=\.)\s+(?=[A-Z][a-z])", tail):
            candidate = item.strip().lstrip("-*").strip()
            # Strip leading numbering (1., 1), (a), etc.).
            candidate = re.sub(
                r"^(?:\d+[.)]|\([a-z0-9]+\)|[a-z]\.)\s*", "", candidate
            )
            candidate = candidate.rstrip(".")
            if len(candidate) < 15 or len(candidate) > 280:
                continue
            # Must contain at least one verb-ish token (rough check: has a
            # word ending in common verb suffixes or a Bloom verb).
            if re.search(
                r"\b(describe|explain|apply|analyze|evaluate|create|identify|"
                r"define|compare|contrast|differentiate|summarize|list|"
                r"interpret|demonstrate|classify|distinguish|solve|design|"
                r"construct|calculate|predict|assess|outline|relate|"
                r"examine|justify|critique|recognize|recall|state)\b",
                candidate, re.IGNORECASE,
            ):
                los.append(candidate)

    # Strategy 2: inline "After reading this chapter you will be able to X, Y,
    # and Z." — rare but occurs. Split the comma-list and emit each.
    if not los:
        for m in _LO_INLINE_LEADIN_RE.finditer(full_text):
            statements = m.group(1).strip()
            parts = re.split(r",\s*(?:and\s+)?|\band\b", statements)
            for p in parts:
                p_clean = p.strip().rstrip(".")
                if 15 <= len(p_clean) <= 280:
                    los.append(p_clean)

    # De-dupe while preserving order.
    seen: set = set()
    uniq: List[str] = []
    for entry in los:
        key = " ".join(entry.lower().split())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(entry)
    return uniq


def extract_misconceptions(full_text: str) -> List[Dict[str, str]]:
    """Extract ``[{"misconception": str, "correction": str}]`` pairs from
    the source text. Returns an empty list when no recognizable pattern
    was found.

    Priority:
      1. ``Misconception: X. Correction: Y.`` pairs (strict shape).
      2. ``Common misconception: X. In fact, Y.`` (corrective lead-in).

    Both keys must be present in the output entry (schema: both required).
    """
    if not full_text:
        return []
    pairs: List[Dict[str, str]] = []

    # Shape 1: paired Misconception / Correction blocks.
    for m in _MISCONCEPTION_CORRECTION_PAIR_RE.finditer(full_text):
        misconception = m.group(1).strip().rstrip(".")
        correction = m.group(2).strip().rstrip(".")
        # Strip trailing Misconception marker (lookahead can leave stray
        # word fragments in correction).
        misconception = re.sub(
            r"\s*(Correction|Misconception)\s*:.*$", "", misconception
        ).strip()
        correction = re.sub(
            r"\s*(Correction|Misconception)\s*:.*$", "", correction
        ).strip()
        if (
            10 < len(misconception) < 400
            and 10 < len(correction) < 600
        ):
            pairs.append({
                "misconception": misconception,
                "correction": correction,
            })

    # Shape 2: "Common misconception: ..." — no paired correction in the
    # source, so we don't emit (schema requires both keys).
    # We intentionally DO NOT fabricate a correction when the source only
    # provides the misconception side. Drop the entry.

    # De-dupe on the misconception statement.
    seen: set = set()
    uniq: List[Dict[str, str]] = []
    for p in pairs:
        key = " ".join(p["misconception"].lower().split())[:80]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def extract_self_check_questions(full_text: str) -> List[Dict[str, Any]]:
    """Extract real exercise/review question stems from the source text.

    Returns a list of ``{"question": str, "bloom_level": str, "options":
    []}`` dicts. ``options`` is empty because the extracted question text
    rarely includes structured answer choices in a parseable shape;
    downstream, generate_course will render the question as an open
    reflection prompt.

    Returns an empty list when no recognizable exercise pattern was found
    — callers MUST NOT fabricate the legacy multi-choice stem placeholder.
    """
    if not full_text:
        return []
    questions: List[Dict[str, Any]] = []
    for m in _EXERCISE_MARKER_RE.finditer(full_text):
        stem = m.group(2).strip().rstrip(".")
        if len(stem) < 15 or len(stem) > 400:
            continue
        bloom_level, _verb = detect_bloom_level(stem)
        questions.append({
            "question": stem,
            "bloom_level": bloom_level or "understand",
            "options": [],
        })
    # De-dupe on the first 80 chars of the stem.
    seen: set = set()
    uniq: List[Dict[str, Any]] = []
    for q in questions:
        key = " ".join(q["question"].lower().split())[:80]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(q)
    return uniq


# ---------------------------------------------------------------------------
# DART HTML parsing
# ---------------------------------------------------------------------------


def _build_article_block_id_map(html: str) -> List[Tuple[int, int, str]]:
    """Return (start, end, block_id) tuples for every ``<article …
    data-dart-block-id="…">`` in ``html``. Used by the heading-fallback
    parser to ground each topic on the enclosing chapter's block id
    when paragraphs live outside ``<section>`` wrappers (Wave 35).
    """
    spans: List[Tuple[int, int, str]] = []
    for match in re.finditer(r"(?is)<article[^>]*>", html):
        block_id_match = _DATA_DART_BLOCK_ID_RE.search(match.group(0))
        if not block_id_match:
            continue
        block_id = block_id_match.group(1).strip()
        close_idx = html.find("</article>", match.end())
        close_end = close_idx + len("</article>") if close_idx >= 0 else len(html)
        spans.append((match.start(), close_end, block_id))
    return spans


def _article_block_for_offset(
    article_spans: List[Tuple[int, int, str]], offset: int,
) -> str:
    for start, end, block_id in article_spans:
        if start <= offset < end:
            return block_id
    return ""


def _parse_html_heading_fallback(
    html: str,
    stem: str,
    chapter_for_offset,
) -> List[Dict[str, Any]]:
    """Section-boundary fallback for DART HTML where paragraphs live
    outside ``<section>`` wrappers.

    Wave 35: on the Bates corpus DART emits ``<section
    data-dart-block-id="…">`` tags that hold only a heading, while the
    1000+ ``<p>`` tags sit directly inside ``<main>``/``<article>``
    between consecutive sections. The primary section-based parser
    returned zero topics on those files, which fired the Wave 32
    CONTENT_GENERATION_EMPTY guard. This fallback walks every
    ``<section>`` opening tag, harvests its block id + heading, and
    grabs the paragraphs between ``</section>`` and the next
    ``<section>`` (or the end of the document) as the topic body. When
    no ``<section>`` tags are present we fall back to a plain
    ``<h2>``/``<h3>`` boundary scan — block-id grounding is then
    unavailable, but at least the content_nonempty guard clears.
    """
    fallback_topics: List[Dict[str, Any]] = []
    article_spans = _build_article_block_id_map(html)

    section_opens = list(re.finditer(r"(?is)<section([^>]*)>", html))
    if section_opens:
        boundaries: List[Tuple[int, int, str, str]] = []
        for match in section_opens:
            opening_tag = match.group(0)
            block_id_match = _DATA_DART_BLOCK_ID_RE.search(opening_tag)
            block_id = (
                block_id_match.group(1).strip() if block_id_match else ""
            )
            close_idx = html.find("</section>", match.end())
            close_end = close_idx + len("</section>") if close_idx >= 0 else match.end()
            inside_body = (
                html[match.end():close_idx] if close_idx >= 0 else ""
            )
            heading_match = _HEADING_RE.search(inside_body)
            heading_raw = heading_match.group(2) if heading_match else ""
            heading = _strip_tags(heading_raw)
            boundaries.append((match.start(), close_end, heading, block_id))
        # close_idx/close_end captured above per iteration.
        boundaries.append((len(html), len(html), "", ""))

        for i in range(len(boundaries) - 1):
            sec_start, sec_close_end, heading, block_id = boundaries[i]
            next_sec_start = boundaries[i + 1][0]
            body_slice = html[sec_close_end:next_sec_start]
            paragraphs_raw = _PARAGRAPH_RE.findall(body_slice)
            paragraphs: List[str] = []
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
            if _is_low_signal_heading(heading):
                continue
            # Prefer the section's own block id; fall back to the
            # enclosing article's block id when the section didn't
            # carry one (keeps per-paragraph grounding non-empty).
            resolved_block_id = block_id or _article_block_for_offset(
                article_spans, sec_start,
            )
            fallback_topics.append({
                "heading": (heading[:120] or f"Section from {stem}"),
                "paragraphs": paragraphs,
                "key_terms": _extract_key_terms(full_text),
                "source_file": stem,
                "word_count": word_count,
                "chapter_id": chapter_for_offset(sec_start),
                "dart_block_ids": [resolved_block_id] if resolved_block_id else [],
                "extracted_lo_statements": [],
                "extracted_misconceptions": [],
                "extracted_questions": [],
            })
        if fallback_topics:
            return fallback_topics

    # No <section> tags — scan headings directly. Source grounding
    # is unavailable but at least we emit non-empty pages.
    heading_spans: List[Tuple[int, int, str]] = []
    for match in _HEADING_OPEN_RE.finditer(html):
        heading_text = _strip_tags(match.group(2))
        heading_spans.append((match.start(), match.end(), heading_text))
    if not heading_spans:
        return fallback_topics
    heading_spans.append((len(html), len(html), ""))
    for i in range(len(heading_spans) - 1):
        start, end, heading = heading_spans[i]
        next_start = heading_spans[i + 1][0]
        body_slice = html[end:next_start]
        paragraphs_raw = _PARAGRAPH_RE.findall(body_slice)
        paragraphs: List[str] = []
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
        if _is_low_signal_heading(heading):
            continue
        article_block_id = _article_block_for_offset(article_spans, start)
        fallback_topics.append({
            "heading": heading[:120] or f"Section from {stem}",
            "paragraphs": paragraphs,
            "key_terms": _extract_key_terms(full_text),
            "source_file": stem,
            "word_count": word_count,
            "chapter_id": chapter_for_offset(start),
            "dart_block_ids": [article_block_id] if article_block_id else [],
            "extracted_lo_statements": [],
            "extracted_misconceptions": [],
            "extracted_questions": [],
        })
    return fallback_topics


def parse_dart_html_files(html_paths: List[Path]) -> List[Dict[str, Any]]:
    """Parse staged DART HTML files into a flat list of topic dicts.

    Each topic dict:
        {
            "heading": str,                       # cleaned section heading
            "paragraphs": List[str],              # cleaned paragraph text
            "key_terms": List[str],               # heuristic key terms
            "source_file": str,                   # file stem for provenance
            "word_count": int,
            "extracted_lo_statements": List[str], # real LO statements from
                                                  # source (empty when absent)
            "extracted_misconceptions": List[Dict[str, str]],
                                                  # real paired m/c entries
            "extracted_questions": List[Dict[str, Any]],
                                                  # real exercise stems
        }

    Sections with < 30 words are skipped (usually metadata headers).
    """
    topics: List[Dict[str, Any]] = []
    # Per-file extracted content (spans all sections in the file). We run
    # the corpus extractors on the whole-document text because LOs /
    # misconceptions / exercises often live in their own section that may
    # be filtered out by the heading filter (e.g. "Chapter Objectives"
    # is blocklisted as a topic title because it's template chrome, but
    # the BULLETS underneath are real LO content).
    file_lo_map: Dict[str, List[str]] = {}
    file_misconception_map: Dict[str, List[Dict[str, str]]] = {}
    file_question_map: Dict[str, List[Dict[str, Any]]] = {}

    for path in html_paths:
        try:
            html = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        stem = path.stem

        # Whole-document text → feed to corpus extractors BEFORE we apply
        # the heading filter (so content hidden under a blocklisted title
        # like "Chapter Objectives" still gets captured).
        whole_text = _strip_tags(html)
        file_lo_map[stem] = extract_learning_objectives(whole_text)
        file_misconception_map[stem] = extract_misconceptions(whole_text)
        file_question_map[stem] = extract_self_check_questions(whole_text)

        # Wave 24: locate <article role="doc-chapter"> wrappers first so we
        # can tag each topic with its owning chapter. Sections outside any
        # article (e.g. pre-Wave-13 DART output) get chapter_id=None.
        chapter_spans: List[Tuple[int, int, str]] = []  # (start, end, chapter_id)
        for idx, match in enumerate(_DOC_CHAPTER_ARTICLE_RE.finditer(html), start=1):
            # Locate the article's own id= attribute, falling back to a
            # synthesized chN based on position in the file.
            opening = html[match.start():match.start() + 200]
            id_match = _ARTICLE_ID_RE.search(opening)
            ch_id = id_match.group(1) if id_match else f"ch{idx}"
            chapter_spans.append((match.start(), match.end(), ch_id))

        def _chapter_for_offset(offset: int) -> Optional[str]:
            for start, end, ch_id in chapter_spans:
                if start <= offset < end:
                    return ch_id
            return None

        # Prefer DART-shaped <section> blocks; fall back to whole-document
        # heading-boundary split when no <section> tags are present. When
        # <section> is present we also capture its byte-offset so chapter
        # assignment can map the section back to its article wrapper, and
        # Wave 27 captures ``data-dart-block-id`` from the opening tag so
        # source-ids carry through into the Courseforge page.
        # Section tuple: (opening_tag, body, start_offset)
        section_matches: List[Tuple[str, str, int]] = []
        for m in _SECTION_RE.finditer(html):
            section_matches.append((m.group(1), m.group(2), m.start()))
        if not section_matches:
            section_matches = [("", html, 0)]

        topics_before_file = len(topics)
        for opening_tag, section_body, section_offset in section_matches:
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

            # Wave 27: harvest DART block-id from the section wrapper.
            # When absent (pre-Wave-12 DART output), leave empty so the
            # downstream renderer falls back to the Wave 9 source-
            # module-map path or silently elides source-ids.
            block_id_match = _DATA_DART_BLOCK_ID_RE.search(opening_tag)
            dart_block_id = (
                block_id_match.group(1).strip() if block_id_match else ""
            )

            topics.append({
                "heading": heading[:120],
                "paragraphs": paragraphs,
                "key_terms": _extract_key_terms(full_text),
                "source_file": stem,
                "word_count": word_count,
                # Wave 24: chapter_id from <article role="doc-chapter">;
                # None when DART didn't emit doc-chapter wrappers.
                "chapter_id": _chapter_for_offset(section_offset),
                # Wave 27: DART provenance for per-element source-id
                # carry-through. ``dart_block_ids`` is a list so future
                # multi-block topics (e.g. merged sibling sections) can
                # contribute a comma-joined source-ids attribute.
                "dart_block_ids": [dart_block_id] if dart_block_id else [],
                # Populated in the finalization pass below so every topic
                # from the same source carries the file's extracted items.
                "extracted_lo_statements": [],
                "extracted_misconceptions": [],
                "extracted_questions": [],
            })

        # Wave 35: when the section-based pass added nothing for this
        # file (DART emitted heading-only <section> tags with the real
        # paragraphs floating in <main>), fall back to a
        # heading-boundary scan over the whole HTML so we still ground
        # Courseforge pages in the source material.
        if len(topics) == topics_before_file:
            topics.extend(
                _parse_html_heading_fallback(html, stem, _chapter_for_offset)
            )

    # De-duplicate topics that share a normalized heading (case- and
    # whitespace-insensitive). Keeps the first occurrence, which tends
    # to be the most content-rich one when a heading repeats across
    # front-matter / index / chapter body.
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for topic in topics:
        key = " ".join(topic["heading"].lower().split())
        if key in seen:
            continue
        seen.add(key)
        stem = topic.get("source_file", "")
        topic["extracted_lo_statements"] = list(
            file_lo_map.get(stem, [])
        )
        topic["extracted_misconceptions"] = [
            dict(m) for m in file_misconception_map.get(stem, [])
        ]
        topic["extracted_questions"] = [
            dict(q) for q in file_question_map.get(stem, [])
        ]
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
    *,
    max_terminal: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate canonical objectives from parsed DART topics.

    Output shape matches what ``generate_week`` consumes. IDs follow the
    JSON-LD schema's ``^[A-Z]{2,}-\\d{2,}$`` pattern: ``TO-NN`` for terminal,
    ``CO-NN`` for chapter-level (minted via
    :func:`lib.ontology.learning_objectives.mint_lo_id`, the Wave 24
    canonical helper — no more inline f-string magic).

    Content policy (NO placeholder generations):
      * Every emitted objective statement MUST come from the source corpus
        — either from an extracted "Learning Objectives" list on the parse
        side, or from a real section heading. No templated phrases
        ("Apply concepts from X to analyze real-world examples.",
        "Describe X and explain the core ideas.", etc.) are ever emitted.
      * When the corpus is empty (no topics at all), this returns empty
        lists. The caller (``_generate_course_content``) handles that by
        emitting pages with an empty objectives array — schema-compliant,
        since ``learningObjectives`` is not required top-level.

    Args:
        topics: List of parsed DART topic dicts.
        duration_weeks: Target course duration in weeks (used for Path B
            heading grouping).
        max_terminal: Terminal-outcome ceiling. Default 2 preserves the
            historical behaviour; :func:`_plan_course_structure` can pass
            a larger value when the corpus is rich enough.
    """
    if not topics:
        # Empty corpus → empty objective lists. No placeholder synthesis.
        return ([], [])

    # Group topics into weeks round-robin — later used for both objective
    # derivation and per-week content binding.
    topics_per_week = _group_topics_by_week(topics, duration_weeks)

    # First: harvest all real LO statements extracted by parse_dart_html_files.
    # A single textbook section may emit multiple LOs; we round-robin assign
    # them across weeks below.
    all_extracted_los: List[Tuple[str, List[str]]] = []  # (heading, [statements])
    for topic in topics:
        statements = topic.get("extracted_lo_statements") or []
        if statements:
            all_extracted_los.append((topic["heading"], statements))

    terminal: List[Dict[str, Any]] = []
    chapter: List[Dict[str, Any]] = []

    to_counter = 1
    co_counter = 1

    # Path A: real LOs were extracted. Emit those verbatim.
    if all_extracted_los:
        for heading, statements in all_extracted_los:
            primary_term_slug = canonical_slug(heading) or ""
            for statement in statements:
                level, verb = detect_bloom_level(statement)
                entry: Dict[str, Any] = {
                    "statement": statement,
                    "key_concepts": [primary_term_slug] if primary_term_slug else [],
                }
                if level:
                    entry["bloom_level"] = level
                if verb:
                    entry["bloom_verb"] = verb
                # First ``max_terminal`` go to terminal; the rest are COs.
                if to_counter <= max_terminal:
                    entry["id"] = mint_lo_id("terminal", to_counter)
                    terminal.append(entry)
                    to_counter += 1
                else:
                    entry["id"] = mint_lo_id("chapter", co_counter)
                    chapter.append(entry)
                    co_counter += 1
        return (terminal, chapter)

    # Path B: no real LOs extracted. Use real heading text as the LO
    # statement. Heading text is literal source material — not fabricated
    # prose. Downstream consumers see "Introduction to Photosynthesis" as
    # the objective statement, which is less pedagogically framed but
    # guaranteed non-placeholder.
    for week_num, week_topics in enumerate(topics_per_week, start=1):
        if not week_topics:
            continue
        primary = week_topics[0]
        primary_heading = primary["heading"]
        primary_terms = primary.get("key_terms") or [primary_heading]
        level, verb = detect_bloom_level(primary_heading)
        # Terminals capped at max_terminal; overflow primaries become COs.
        if to_counter <= max_terminal:
            terminal_entry: Dict[str, Any] = {
                "id": mint_lo_id("terminal", to_counter),
                "statement": primary_heading,
                "key_concepts": [canonical_slug(t) for t in primary_terms[:3]
                                 if canonical_slug(t)],
            }
            if level:
                terminal_entry["bloom_level"] = level
            if verb:
                terminal_entry["bloom_verb"] = verb
            terminal.append(terminal_entry)
            to_counter += 1
        else:
            primary_entry: Dict[str, Any] = {
                "id": mint_lo_id("chapter", co_counter),
                "statement": primary_heading,
                "key_concepts": [canonical_slug(t) for t in primary_terms[:3]
                                 if canonical_slug(t)],
            }
            if level:
                primary_entry["bloom_level"] = level
            if verb:
                primary_entry["bloom_verb"] = verb
            chapter.append(primary_entry)
            co_counter += 1

        # One CO per additional heading in the week (chapter-level
        # objectives bind to the secondary sections the week covers).
        for secondary in week_topics[1:]:
            sec_heading = secondary["heading"]
            sec_terms = secondary.get("key_terms") or [sec_heading]
            sec_level, sec_verb = detect_bloom_level(sec_heading)
            chapter_entry: Dict[str, Any] = {
                "id": mint_lo_id("chapter", co_counter),
                "statement": sec_heading,
                "key_concepts": [canonical_slug(t) for t in sec_terms[:3]
                                 if canonical_slug(t)],
            }
            if sec_level:
                chapter_entry["bloom_level"] = sec_level
            if sec_verb:
                chapter_entry["bloom_verb"] = sec_verb
            chapter.append(chapter_entry)
            co_counter += 1

    return (terminal, chapter)


def _page_roles_for_week(lo_count: int) -> Tuple[str, ...]:
    """Return the canonical page-role tuple for a week with ``lo_count`` LOs.

    Wave 24 HIGH-5 fix: pre-Wave-24, ``pipeline_tools.py`` hardcoded a
    5-tuple (overview, content_01, application, self_check, summary) for
    every week regardless of how many LOs the week carried. That meant a
    1-LO week got 5 pages (mostly filler) and a 12-LO week also got 5
    pages (one content page cramming 12 LOs). This helper scales the
    content-page count with ``lo_count``:

      * 1 ``overview`` page (always)
      * ⌈lo_count / 2⌉ ``content_NN`` pages (min 1, max 6 content pages)
      * 1 ``application`` page
      * 1 ``self_check`` page
      * 1 ``summary`` page

    Total is clamped to [3, 10] — the floor keeps the 5-page test
    fixtures that still depend on the old minimum alive; the ceiling
    avoids pathologically-long weeks.
    """
    if lo_count < 0:
        lo_count = 0
    content_count = max(1, (lo_count + 1) // 2)
    content_count = min(content_count, 6)

    roles: List[str] = ["overview"]
    for i in range(1, content_count + 1):
        roles.append(f"content_{i:02d}")
    roles.extend(["application", "self_check", "summary"])

    # Floor: minimum 3 pages (overview + content + summary). Ceiling: 10.
    if len(roles) < 3:
        # Defensive — shouldn't hit given overview + 1 content + 3 tail = 5.
        roles = ["overview", "content_01", "summary"]
    if len(roles) > 10:
        # Trim content_NN tail while preserving tail labels.
        tail = ["application", "self_check", "summary"]
        head = roles[: 10 - len(tail)]
        roles = head + tail
    return tuple(roles)


def _group_topics_by_week(
    topics: List[Dict[str, Any]],
    duration_weeks: int,
    *,
    max_topics_per_week: int = 12,
) -> List[List[Dict[str, Any]]]:
    """Return a list of length ``duration_weeks``; each entry is the list
    of topics assigned to that week.

    Wave 24: when DART emits ``<article role="doc-chapter">`` wrappers,
    the parser tags every topic with a ``chapter_id``. We prefer to keep
    all topics from the same chapter in the same week (so the week
    aligns with a real textbook chapter). Chapters are only split across
    weeks when they exceed ``max_topics_per_week``. When no chapter_ids
    are present (pre-Wave-13 DART output) we fall back to the legacy
    positional bucketing so older fixtures don't regress.

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

    # Wave 24: prefer chapter-respecting grouping when chapter_ids are
    # present on every topic. If any topic lacks a chapter_id, fall back
    # to legacy positional bucketing so mixed corpora don't lose topics.
    has_chapters = all(
        bool(t.get("chapter_id")) for t in topics
    )
    if has_chapters:
        # Preserve insertion order of chapters as they appear in the corpus.
        chapter_order: List[str] = []
        chapter_topics: Dict[str, List[Dict[str, Any]]] = {}
        for t in topics:
            cid = t["chapter_id"]
            if cid not in chapter_topics:
                chapter_order.append(cid)
                chapter_topics[cid] = []
            chapter_topics[cid].append(t)

        # Flatten each chapter into week-sized pieces (splitting only
        # when > max_topics_per_week). Then distribute pieces across
        # ``duration_weeks`` buckets in order, never assigning two
        # different chapters to the same bucket when buckets remain.
        pieces: List[List[Dict[str, Any]]] = []
        for cid in chapter_order:
            ch_topics = chapter_topics[cid]
            if len(ch_topics) <= max_topics_per_week:
                pieces.append(ch_topics)
            else:
                # Split into ceil(len/max) pieces of roughly equal size.
                step = max(1, (len(ch_topics) + max_topics_per_week - 1)
                           // max_topics_per_week)
                piece_count = (len(ch_topics) + step - 1) // step
                for i in range(piece_count):
                    pieces.append(ch_topics[i * step:(i + 1) * step])

        # Assign pieces to buckets round-robin, one piece per bucket
        # when possible. When pieces > duration_weeks, later pieces pile
        # into the tail bucket (preserves all topics; better than dropping).
        for idx, piece in enumerate(pieces):
            week_idx = min(idx, duration_weeks - 1)
            buckets[week_idx].extend(piece)
        return buckets

    # Legacy positional bucketing — retained for pre-Wave-13 DART output
    # and non-DART HTML that doesn't carry chapter_ids.
    per_week = max(1, (len(topics) + duration_weeks - 1) // duration_weeks)
    for idx, topic in enumerate(topics):
        week_idx = min(idx // per_week, duration_weeks - 1)
        buckets[week_idx].append(topic)
    return buckets


# ---------------------------------------------------------------------------
# Per-week week_data assembly
# ---------------------------------------------------------------------------


# Bloom-level -> apply-phase prompt verb. Keeps the per-week prompt grounded
# in the week's own cognitive demand rather than defaulting every activity
# to "demonstrate the concept." ``analyze`` / ``evaluate`` weeks should ask
# the student to compare / critique, not just restate.
_BLOOM_APPLY_VERB = {
    "remember": "recall",
    "understand": "explain",
    "apply": "apply",
    "analyze": "compare",
    "evaluate": "evaluate",
    "create": "design",
}


def _build_activity_prompt(
    *,
    week_title: str,
    week_topics: List[Dict[str, Any]],
    week_objectives: List[Dict[str, Any]],
    first_obj_statement: str,
) -> Tuple[str, str]:
    """Assemble a per-week activity prompt description + Bloom level.

    Policy:
      * Prompt references the week's **own** key terms when any topic
        exposed them via ``_extract_key_terms`` / DART heading analysis.
      * Prompt chooses an action verb based on the first objective's
        Bloom level so an ``analyze`` week doesn't get a ``demonstrate``
        prompt. Falls back to ``apply`` when no Bloom signal is present.
      * When no topic or key-term data exists (empty-corpus week), emits
        a neutral prompt keyed off the objective statement — NO
        tautological "the concept from the week's material" tail.

    Returns ``(description, bloom_level)``. ``description`` is un-escaped
    raw text; the caller is responsible for ``_html.escape`` before
    inserting into HTML.
    """
    # Harvest up to 2 distinctive key terms across the week's topics.
    seen_terms: set = set()
    terms: List[str] = []
    for topic in week_topics or []:
        for term in topic.get("key_terms") or []:
            key = term.lower().strip()
            if not key or key in seen_terms:
                continue
            seen_terms.add(key)
            terms.append(term)
            if len(terms) >= 2:
                break
        if len(terms) >= 2:
            break

    # Pick a Bloom-level-aware verb for the prompt's call-to-action.
    bloom_level = (
        (week_objectives[0].get("bloom_level") if week_objectives else None)
        or "apply"
    )
    verb = _BLOOM_APPLY_VERB.get(bloom_level, "apply")

    objective_stem = first_obj_statement.rstrip(".").strip()

    if terms and week_topics:
        # Prefer a prompt that names the actual terminology from the
        # week's source material. Example: "Drawing on the week's
        # reading, compare *domain_knowledge* and *procedural_knowledge*
        # in light of the learning objective: 'Differentiate ...'."
        term_list = ", ".join(terms)
        description = (
            f"Drawing on this week's reading, {verb} "
            f"{term_list} in the context of the learning objective: "
            f"\"{objective_stem}.\" "
            f"Respond in roughly 150 words, citing at least one "
            f"specific passage or example from the assigned material."
        )
    elif week_topics:
        # Topic exists but no clean key terms — fall back to the topic
        # heading as the anchor instead of the boilerplate phrase.
        topic_heading = week_topics[0].get("heading") or week_title
        description = (
            f"Drawing on the section \"{topic_heading}\", {verb} the "
            f"ideas behind the learning objective: "
            f"\"{objective_stem}.\" Respond in roughly 150 words and "
            f"support your answer with one example from the reading."
        )
    else:
        # No topic data at all: neutral prompt, no "demonstrating the
        # concept from the week's material" tail.
        description = (
            f"Working from the week's reading, {verb} the ideas behind "
            f"the learning objective: \"{objective_stem}.\" "
            f"Respond in roughly 150 words using your own examples."
        )

    return description, ("apply" if verb == "apply" else bloom_level)


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
    ``tests/fixtures/pipeline/reference_week_01/``. We produce:
      * one ``overview`` page (from the first topic / week heading),
      * **N** ``content`` modules — **dynamic**, one per LO / distinct
        source topic. Minimum 1 to preserve the 5-page floor.
      * one ``application`` activity,
      * one ``self_check`` quiz (only when real questions extracted),
      * one ``summary``.

    Content-policy note: this builder emits only grounded content —
    heading + paragraph text pulled from the DART-staged HTML, plus
    source-extracted misconceptions / self-check questions. When no real
    extraction is available, the relevant section emits an empty list
    (misconceptions, self_check_questions) so downstream consumers see a
    schema-clean absence rather than templated filler.
    """
    primary_topic = week_topics[0] if week_topics else None

    if primary_topic:
        week_title = primary_topic["heading"]
    else:
        # Fallback when no topic is bound to this week. The emitter in
        # ``generate_week`` wraps this as ``"Week {N} Overview: {title}"``,
        # so a neutral label here avoids the tautological
        # ``"Week N Overview: Week N Concepts"`` H1 observed on corpora
        # where week count exceeds topic count.
        week_title = "Overview"

    # Overview: week-level paragraphs + readings
    overview_text: List[str] = []
    if primary_topic and primary_topic["paragraphs"]:
        overview_text.append(primary_topic["paragraphs"][0])
        if len(primary_topic["paragraphs"]) > 1:
            overview_text.append(primary_topic["paragraphs"][1])
    elif week_topics:
        # No primary with paragraphs but other topics exist — use their
        # text so overview carries real source content.
        for t in week_topics:
            if t.get("paragraphs"):
                overview_text.append(t["paragraphs"][0])
                break
    # If overview_text is STILL empty, we leave it empty — generate_week
    # renders an overview heading + objectives list regardless.

    # ---------------------------------------------------------------- #
    # Dynamic content modules: one per LO when LOs are rich, otherwise
    # one per distinct source topic. Minimum 1 to preserve the 5-page
    # floor required by the integration-test contract.
    # ---------------------------------------------------------------- #
    content_modules = _build_content_modules_dynamic(
        week_topics=week_topics,
        week_objectives=week_objectives,
        week_title=week_title,
    )

    # Activities — one practice activity tied to first objective. The
    # description quotes the real objective statement (extracted) when
    # available; otherwise falls back to the week title (real heading).
    activity_objective_ref = (
        week_objectives[0]["id"] if week_objectives else None
    )
    first_obj_statement = (
        week_objectives[0]["statement"] if week_objectives else week_title
    )
    # Per-week activity description — varies by topic/key-terms/bloom so
    # weeks don't emit a copy-pasted identical prompt body. Falls back to
    # a neutral wording when the week has no topic data to ground on.
    activity_description, activity_bloom = _build_activity_prompt(
        week_title=week_title,
        week_topics=week_topics,
        week_objectives=week_objectives,
        first_obj_statement=first_obj_statement,
    )
    activities = [{
        "title": f"Apply: {week_title}",
        "description": _html.escape(activity_description),
        "bloom_level": activity_bloom,
        **({"objective_ref": activity_objective_ref}
           if activity_objective_ref else {}),
    }]

    # Self-check questions — extracted from source; empty when none.
    self_check_questions = _build_self_check_questions(
        week_topics, week_objectives
    )

    # Summary key takeaways — from real topic headings only.
    key_takeaways: List[str] = []
    seen_takeaway: set = set()
    for t in week_topics:
        heading = t.get("heading", "")
        key = heading.lower().strip()
        if not key or key in seen_takeaway:
            continue
        seen_takeaway.add(key)
        key_takeaways.append(heading)
    # Omit the "Map each learning objective…" boilerplate takeaway. When
    # no topics are present, key_takeaways is empty — generate_week
    # handles that by emitting the heading only.

    # Reflection questions: when real objectives exist, echo their
    # statement as a reflection prompt (not a fabricated stem).
    reflection_questions: List[str] = []
    for obj in (week_objectives or [])[:2]:
        statement = obj.get("statement", "").strip().rstrip(".")
        if statement:
            reflection_questions.append(
                f"Restate in your own words: {statement}."
            )

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


def _build_content_modules_dynamic(
    week_topics: List[Dict[str, Any]],
    week_objectives: List[Dict[str, Any]],
    week_title: str,
) -> List[Dict[str, Any]]:
    """Return ``content_modules`` list — **one module per LO or topic**.

    Policy (per user directive — "number of html files per week should be
    dynamic based on learning objectives identified"):
      * N = max(len(week_objectives), len(week_topics), 1) — when we
        have 3 distinct LOs we want 3 content pages, each focused on one
        LO + its source section.
      * When objectives and topics exist in different counts, we pair
        them positionally: topic[i] is the source material for
        objective[i] (when both indices exist).
      * Module title = the topic heading (real source text) or the LO
        statement (real source text) — never fabricated.
      * Module sections = the topic's paragraphs. When no topic is
        available for position ``i`` but an LO is, we fall back to a
        single minimal section with the LO statement as heading; this is
        literal source content, not a placeholder.

    Minimum one module to preserve the integration test's 5-page floor.
    """
    # Per-file misconceptions. When a topic has extracted misconceptions,
    # those attach to the module drawing from that topic.
    modules: List[Dict[str, Any]] = []
    topic_count = len(week_topics)
    obj_count = len(week_objectives)
    module_count = max(topic_count, obj_count, 1)

    for i in range(module_count):
        topic = week_topics[i] if i < topic_count else None
        obj = week_objectives[i] if i < obj_count else None

        # Title selection: prefer real topic heading; fall back to the
        # LO statement (truncated) when no topic at this index.
        if topic:
            module_title = topic["heading"]
        elif obj:
            module_title = obj.get("statement") or week_title
            module_title = module_title[:120]
        else:
            module_title = week_title

        # Sections: built from the topic's paragraphs. If no topic at
        # this index, emit a minimal section whose heading is the LO
        # statement (source content).
        if topic:
            section_role = "definition" if i == 0 else "explanation"
            sections = [_topic_to_section(topic, section_role=section_role)]
        elif obj:
            sections = [{
                "heading": obj.get("statement", "")[:120] or week_title,
                "level": 2,
                "content_type": "explanation",
                "paragraphs": [],
                "key_terms": [],
            }]
        else:
            # True empty corpus: emit a placeholder-free minimal section.
            sections = [{
                "heading": week_title,
                "level": 2,
                "content_type": "explanation",
                "paragraphs": [],
                "key_terms": [],
            }]

        # Per-module misconceptions come from the linked topic when present.
        module_misconceptions = (
            list(topic.get("extracted_misconceptions") or [])
            if topic else []
        )

        modules.append({
            "title": module_title,
            "sections": sections,
            "misconceptions": module_misconceptions,
        })

    return modules


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

    Wave 27 HIGH-3: when the DART source carries ``data-dart-block-id``
    on the section wrapper, emit ``source_references[]`` on the
    generated section so
    :func:`Courseforge.scripts.generate_course._render_content_sections`
    stamps ``data-cf-source-ids="dart:{slug}#{block_id}"`` on the
    rendered ``<h2>`` wrapper. Also propagates into the page's JSON-LD
    ``sections[].sourceReferences`` via ``_build_section_metadata``.
    Back-compat: when DART didn't emit block IDs, the refs list is
    empty and nothing is stamped.
    """
    key_terms = topic.get("key_terms", []) or []
    paragraphs = [_html.escape(p) for p in topic["paragraphs"][:3]]
    section: Dict[str, Any] = {
        "heading": topic["heading"],
        "level": 2,
        "content_type": section_role,
        "paragraphs": paragraphs,
        "key_terms": key_terms,
    }
    source_refs = _topic_source_references(topic)
    if source_refs:
        section["source_references"] = source_refs
    return section


def _topic_source_references(
    topic: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build a ``sourceReferences[]`` list from a parsed DART topic.

    Wave 27: every block ID captured on the source <section> wrapper
    becomes one ``{sourceId, role}`` entry. The first block ID plays
    the ``primary`` role; any additional IDs (future: multi-block
    topics merged into one) play ``contributing``. Returns an empty
    list when the topic has no DART block IDs — back-compat path for
    pre-Wave-12 DART HTML.

    Slug normalization differs between the two halves of the
    ``dart:{slug}#{block_id}`` shape:

    * Document slug — Wave 35 switched from :func:`canonical_slug`
      (which collapses underscores into one token) to a gentler
      lowercase + space-to-hyphen transform that matches the
      :class:`ContentGroundingValidator` and Wave 9 source-router.
      Pre-Wave-35 emitted slugs like ``batesteachingdigitalageaccessible``
      couldn't resolve against validator-visible staged HTML whose
      stem was ``bates_teaching_digital_age_accessible``.
    * Block ID — uses a gentler lowercase + pattern-filter so DART's
      native ``s3_c0`` / 16-hex IDs survive unchanged (the
      ``canonical_slug`` helper would collapse underscores and break
      the schema pattern).
    """
    block_ids = [
        bid.strip() for bid in (topic.get("dart_block_ids") or [])
        if isinstance(bid, str) and bid.strip()
    ]
    if not block_ids:
        return []
    stem = topic.get("source_file") or ""
    if not stem:
        return []
    slug = stem.lower().replace(" ", "-")
    refs: List[Dict[str, Any]] = []
    for idx, block_id in enumerate(block_ids):
        # The source_reference schema requires block_id to match
        # ``[a-z0-9_-]+``. Lowercase + strip any characters outside
        # that set, keeping underscores and hyphens intact (DART's
        # native ``s3_c0`` positional IDs MUST survive unchanged).
        block_slug = re.sub(r"[^a-z0-9_-]+", "", block_id.lower())
        if not block_slug:
            continue
        refs.append({
            "sourceId": f"dart:{slug}#{block_slug}",
            "role": "primary" if idx == 0 else "contributing",
        })
    return refs


def _build_self_check_questions(
    week_topics: List[Dict[str, Any]],
    week_objectives: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return real self-check questions extracted from the source corpus.

    Policy: only emits entries that came from real exercise / review-question
    markers in the DART-staged HTML (see :func:`extract_self_check_questions`).
    When no real questions were extracted for any of the week's topics, returns
    an empty list — downstream ``generate_week`` then skips the self_check
    page entirely. The legacy multi-choice stem placeholder is never produced
    here.

    Each returned question dict carries the canonical keys
    ``{"question", "bloom_level", "options", "objective_ref"}``. ``options``
    is an empty list (extracted exercises rarely include structured choices
    in a parseable shape); the self-check page renders open-ended prompts
    in that case. ``objective_ref`` is attached when a positional LO is
    available for the question's index.
    """
    questions: List[Dict[str, Any]] = []
    if not week_topics:
        return []

    # Collect all extracted questions across the week's topics.
    for topic in week_topics:
        for q in topic.get("extracted_questions") or []:
            # Defensive copy + normalize shape.
            entry: Dict[str, Any] = {
                "question": q["question"],
                "bloom_level": q.get("bloom_level") or "understand",
                "options": list(q.get("options") or []),
            }
            questions.append(entry)

    # Bind each question to an objective by position (stable for canonical
    # IDs). Missing positions drop the objective_ref key — the self-check
    # page still validates (objective_ref is optional in generate_week).
    for idx, q in enumerate(questions):
        if idx < len(week_objectives):
            q["objective_ref"] = week_objectives[idx]["id"]

    return questions


def _build_misconceptions_for_week(
    week_topics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return real misconception/correction pairs extracted from the source.

    Policy: only emits entries produced by :func:`extract_misconceptions`
    (strict ``Misconception: ... / Correction: ...`` shape in the DART
    text). Returns an empty list when no real pairs were extracted for any
    of the week's topics — no "Students often assume X is a single idea"
    template is ever produced here.

    Output conforms to the ``Misconception`` JSON-LD schema: each dict has
    both ``misconception`` and ``correction`` keys populated with strings.
    """
    if not week_topics:
        return []
    merged: List[Dict[str, str]] = []
    seen: set = set()
    for topic in week_topics:
        for m in topic.get("extracted_misconceptions") or []:
            mis = m.get("misconception", "").strip()
            cor = m.get("correction", "").strip()
            if not mis or not cor:
                continue
            key = " ".join(mis.lower().split())[:80]
            if key in seen:
                continue
            seen.add(key)
            merged.append({"misconception": mis, "correction": cor})
    return merged


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
    source_ids: Optional[List[str]] = None,
    source_primary: Optional[str] = None,
) -> str:
    """Render a ``<section id="objectives">`` block with per-objective
    ``data-cf-objective-id`` / ``data-cf-bloom-*`` attributes.

    Mirrors the reference_week_01 fixture shape so every emitted page
    has a discoverable objectives surface (the ``page_objectives`` gate
    scans for ``data-cf-objective-id`` on every page, not just overview).

    Wave 35: optional ``source_ids`` stamp ``data-cf-source-ids`` on the
    outer ``<section>`` so :class:`ContentGroundingValidator`'s ancestor
    walk can ground the ``<li>`` items (some synthesized LO statements
    exceed the 30-word non-trivial floor).
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
    source_attrs = ""
    if source_ids:
        joined = ",".join(_html.escape(sid) for sid in source_ids if sid)
        if joined:
            source_attrs = f' data-cf-source-ids="{joined}"'
            if source_primary:
                source_attrs += (
                    f' data-cf-source-primary="{_html.escape(source_primary)}"'
                )
    return (
        f'\n    <section id="objectives" class="objectives" '
        f'aria-labelledby="objectives-heading"{source_attrs}>\n'
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

    Wave 35: when the page body carries a ``<section
    data-cf-source-ids="…">`` wrapper (emitted by
    ``_render_content_sections`` on content pages), mirror those ids
    onto the injected objectives ``<section>`` so
    :class:`ContentGroundingValidator`'s ancestor walk finds grounding
    on long LO statements. No-op on pages without page-level grounding.
    """
    if _OBJECTIVES_SENTINEL in html_text:
        return html_text
    if "data-cf-objective-id" in html_text:
        return html_text
    src_match = re.search(
        r'(?is)<section[^>]*\bdata-cf-source-ids\s*=\s*"([^"]+)"[^>]*>',
        html_text,
    )
    src_ids: Optional[List[str]] = None
    src_primary: Optional[str] = None
    if src_match:
        src_ids = [s.strip() for s in src_match.group(1).split(",") if s.strip()]
        primary_match = re.search(
            r'(?is)data-cf-source-primary\s*=\s*"([^"]+)"',
            src_match.group(0),
        )
        if primary_match:
            src_primary = primary_match.group(1).strip()
    section = _render_objectives_section(objectives, source_ids=src_ids, source_primary=src_primary)
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
    "extract_learning_objectives",
    "extract_misconceptions",
    "extract_self_check_questions",
]
